import builtins
import functools
import inspect
import itertools
import logging
import sys
import textwrap
import time
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type, Union
from unittest.mock import patch
import sympy
import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, identity, preserve_rng_state
from . import config, ir
from .autotune_process import TensorMeta, TritonBenchmarkRequest
from .codecache import code_hash, PersistentCache, PyCodeCache
from .codegen.common import ChoiceCaller, IndentedBuffer, KernelTemplate
from .codegen.triton import texpr, TritonKernel, TritonPrinter, TritonScheduling
from .codegen.triton_utils import config_of, signature_to_meta
from .exc import CUDACompileError
from .utils import do_bench, Placeholder, sympy_dot, sympy_product, unique
from .virtualized import V
from . import lowering  # noqa: F401
class TritonTemplateKernel(TritonKernel):

    def __init__(self, kernel_name, input_nodes, output_node, defines, num_stages, num_warps, grid_fn, meta, call_sizes, use_jit=True, prefix_args=0, suffix_args=0, epilogue_fn=identity, *, index_dtype):
        super().__init__(sympy_product(output_node.get_size()), sympy.Integer(1), index_dtype=index_dtype)
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.named_input_nodes = {}
        self.defines = defines
        self.kernel_name = kernel_name
        self.template_mask = None
        self.use_jit = use_jit
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.grid_fn = grid_fn
        self.meta = meta
        self.call_sizes = call_sizes
        self.prefix_args = prefix_args
        self.suffix_args = suffix_args
        self.epilogue_fn = epilogue_fn
        self.render_hooks = dict()

    def need_numel_args(self):
        return False

    def jit_line(self):
        if self.use_jit:
            return '@triton.jit'
        argdefs, _, signature = self.args.python_argdefs()
        triton_meta = {'signature': signature_to_meta(signature, size_dtype=self.index_dtype), 'device': V.graph.scheduler.current_device.index, 'device_type': V.graph.scheduler.current_device.type, 'constants': {}}
        triton_meta['configs'] = [config_of(signature)]
        inductor_meta = {'kernel_name': str(Placeholder.DESCRIPTIVE_NAME)}
        return textwrap.dedent(f'\n            @template(\n                num_stages={self.num_stages},\n                num_warps={self.num_warps},\n                triton_meta={triton_meta!r},\n                inductor_meta={inductor_meta!r},\n            )\n            @triton.jit\n            ')

    def def_kernel(self, *argnames):
        """
        Hook called from template code to generate function def and
        needed args.
        """
        assert all((isinstance(x, str) for x in argnames))
        renames = IndentedBuffer(initial_indent=1)
        named_args = self.input_nodes[self.prefix_args:len(self.input_nodes) - self.suffix_args]
        assert len(argnames) == len(named_args), (len(argnames), len(named_args), self.prefix_args, len(self.input_nodes))
        for input_node in self.input_nodes[:self.prefix_args]:
            self.args.input(input_node.get_name())
        for name, input_node in zip(argnames, named_args):
            arg_name = f'arg_{name}'
            self.named_input_nodes[name] = input_node
            self.args.input_buffers[input_node.get_name()] = arg_name
        for name in argnames:
            input_node = self.named_input_nodes[name]
            arg_name = self.args.input_buffers[input_node.get_name()]
            if input_node.get_layout().offset == 0:
                renames.writeline(f'{name} = {arg_name}')
            else:
                offset = texpr(self.rename_indexing(input_node.get_layout().offset))
                renames.writeline(f'{name} = {arg_name} + {offset}')
        for input_node in self.input_nodes[len(self.input_nodes) - self.suffix_args:]:
            self.args.input(input_node.get_name())

        def hook():
            arg_defs, *_ = self.args.python_argdefs()
            return '\n'.join(['import triton.language as tl', 'import triton', 'from torch._inductor.triton_heuristics import template', 'from torch._inductor.utils import instance_descriptor', 'from torch._inductor import triton_helpers', '', self.jit_line(), f'def {self.kernel_name}({', '.join(arg_defs)}):', self.defines, renames.getvalue()])
        assert '<DEF_KERNEL>' not in self.render_hooks
        self.render_hooks['<DEF_KERNEL>'] = hook
        return '<DEF_KERNEL>'

    def size(self, name: str, index: int):
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
        assert isinstance(index, int)
        if name is None:
            val = self.output_node.get_size()[index]
        else:
            assert isinstance(name, str)
            val = self.named_input_nodes[name].get_size()[index]
        return texpr(self.rename_indexing(val))

    def stride(self, name, index):
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
        assert isinstance(index, int)
        if name is None:
            val = self.output_node.get_stride()[index]
        else:
            assert isinstance(name, str)
            val = self.named_input_nodes[name].get_stride()[index]
        return texpr(self.rename_indexing(val))

    def store_output(self, indices, val, mask):
        """
        Hook called from template code to store the final output
        (if the buffer hasn't been optimized away), then append any
        epilogue fusions.
        """
        assert isinstance(indices, (list, tuple))
        assert isinstance(val, str)
        assert isinstance(mask, str)
        assert self.template_mask is None
        indices = list(map(TritonPrinter.paren, indices))
        index_symbols = [sympy.Symbol(x) for x in indices]
        lengths = [V.graph.sizevars.simplify(s) for s in self.output_node.get_size()]
        assert len(indices) == len(lengths)
        for name, range_tree_entry in zip(indices, self.range_trees[0].construct_entries(lengths)):
            range_tree_entry.set_name(name)
        contiguous_index = sympy_dot(ir.FlexibleLayout.contiguous_strides(lengths), index_symbols)
        contiguous_index = self.rename_indexing(contiguous_index)
        self.body.writeline('xindex = ' + texpr(contiguous_index))
        self.range_trees[0].lookup(sympy.Integer(1), sympy_product(lengths)).set_name('xindex')
        self.template_mask = mask
        self.template_indices = indices
        output_index = self.output_node.get_layout().make_indexer()(index_symbols)
        output_index = self.rename_indexing(output_index)
        if output_index == contiguous_index:
            output_index = sympy.Symbol('xindex')
        epilogue_args = [val]
        for input_node in itertools.chain(self.input_nodes[:self.prefix_args], self.input_nodes[len(self.input_nodes) - self.suffix_args:]):
            input_node.freeze_layout()
            epilogue_args.append(input_node.make_loader()(index_symbols))
        V.ops.store(self.output_node.get_name(), output_index, self.epilogue_fn(*epilogue_args))
        self.codegen_body()

        def hook():
            self.codegen_body()
            return textwrap.indent(self.body.getvalue(), '    ').strip()
        assert '<STORE_OUTPUT>' not in self.render_hooks
        self.render_hooks['<STORE_OUTPUT>'] = hook
        return '<STORE_OUTPUT>'

    def render(self, template, kwargs):
        return PartialRender(template.render(**self.template_env(), **kwargs), self.render_hooks)

    def make_load(self, name, indices, mask):
        """
        Optional helper called from template code to generate the code
        needed to load from an tensor.
        """
        assert isinstance(indices, (list, tuple))
        assert isinstance(name, str)
        assert isinstance(mask, str)
        stride = self.named_input_nodes[name].get_stride()
        indices = list(map(TritonPrinter.paren, indices))
        assert len(indices) == len(stride)
        index = ' + '.join((f'{texpr(self.rename_indexing(s))} * {i}' for s, i in zip(stride, indices)))
        return f'tl.load({name} + ({index}), {mask})'

    def template_env(self):
        """
        Generate the namespace visible in the template.
        """
        return {fn.__name__: fn for fn in [self.def_kernel, self.size, self.stride, self.store_output, self.make_load]}

    def indexing(self, index: sympy.Expr, *, copy_shape=None, dense_indexing=False, override_mask=None):
        """
        Override the default indexing to use our custom mask and force
        dense indexing.
        """
        result, *mask = super().indexing(index, dense_indexing=False, copy_shape=self.template_mask, override_mask=self.template_mask)
        return (result, *mask)

    def initialize_range_tree(self, pid_cache):
        super().initialize_range_tree(pid_cache)
        self.body.clear()
        self.indexing_code.clear()

    def call_kernel(self, name: str, node: Optional[ir.IRNode]=None):
        wrapper = V.graph.wrapper_code
        _, call_args, _ = self.args.python_argdefs()
        call_args = [str(a) for a in call_args]
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + '.item()'
            if isinstance(call_args[i], sympy.Symbol):
                call_args[i] = texpr(call_args[i])
        if V.graph.cpp_wrapper:
            grid_args = [V.graph.sizevars.simplify(s) for s in self.call_sizes] + [self.meta]
            grid = self.grid_fn(*grid_args)
            wrapper.generate_kernel_call(name, call_args, device_index=V.graph.scheduler.current_device.index, grid=grid)
        else:
            stream_name = wrapper.write_get_raw_stream(V.graph.scheduler.current_device.index)
            wrapper.add_import_once(f'import {self.grid_fn.__module__}')
            meta = wrapper.add_meta_once(self.meta)
            grid_call = [texpr(V.graph.sizevars.simplify(s)) for s in self.call_sizes] + [meta]
            grid_call = f'{self.grid_fn.__module__}.{self.grid_fn.__name__}({', '.join(grid_call)})'
            wrapper.writeline(f'{name}.run({', '.join(call_args)}, grid={grid_call}, stream={stream_name})')