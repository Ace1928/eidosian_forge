import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
def _gen_python_code(self, nodes, root_module: str, namespace: _Namespace, *, verbose: bool=False) -> PythonCode:
    free_vars: List[str] = []
    body: List[str] = []
    globals_: Dict[str, Any] = {}
    wrapped_fns: Dict[str, None] = {}
    maybe_return_annotation: List[str] = ['']

    def add_global(name_hint: str, obj: Any):
        """Add an obj to be tracked as a global.

            We call this for names that reference objects external to the
            Graph, like functions or types.

            Returns: the global name that should be used to reference 'obj' in generated source.
            """
        if _is_from_torch(obj) and obj != torch.device:
            return _get_qualified_name(obj)
        global_name = namespace.create_name(name_hint, obj)
        if global_name in globals_:
            assert globals_[global_name] is obj
            return global_name
        globals_[global_name] = obj
        return global_name
    for name, (_, obj) in _custom_builtins.items():
        add_global(name, obj)

    def type_repr(o: Any):
        if o == ():
            return '()'
        typename = _type_repr(o)
        if hasattr(o, '__origin__'):
            origin_type = _origin_type_map.get(o.__origin__, o.__origin__)
            origin_typename = add_global(_type_repr(origin_type), origin_type)
            if hasattr(o, '__args__'):
                args = [type_repr(arg) for arg in o.__args__]
                if len(args) == 0:
                    return origin_typename
                return f'{origin_typename}[{','.join(args)}]'
            else:
                return origin_typename
        return add_global(typename, o)

    def _get_repr(arg: Any) -> str:
        if isinstance(arg, tuple) and hasattr(arg, '_fields'):
            qualified_name = _get_qualified_name(type(arg))
            global_name = add_global(qualified_name, type(arg))
            return f'{global_name}{repr(tuple(arg))}'
        elif isinstance(arg, torch._ops.OpOverload):
            qualified_name = _get_qualified_name(arg)
            global_name = add_global(qualified_name, arg)
            return f'{global_name}'
        elif isinstance(arg, enum.Enum):
            cls = arg.__class__
            clsname = add_global(cls.__name__, cls)
            return f'{clsname}.{arg.name}'
        return repr(arg)

    def _format_args(args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> str:
        args_s = ', '.join((_get_repr(a) for a in args))
        kwargs_s = ', '.join((f'{k} = {_get_repr(v)}' for k, v in kwargs.items()))
        if args_s and kwargs_s:
            return f'{args_s}, {kwargs_s}'
        return args_s or kwargs_s
    node_to_last_use: Dict[Node, Node] = {}
    user_to_last_uses: Dict[Node, List[Node]] = {}

    def register_last_uses(n: Node, user: Node):
        if n not in node_to_last_use:
            node_to_last_use[n] = user
            user_to_last_uses.setdefault(user, []).append(n)
    for node in reversed(nodes):
        map_arg(node.args, lambda n: register_last_uses(n, node))
        map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    def delete_unused_values(user: Node):
        """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
        if user.op == 'placeholder':
            return
        if user.op == 'output':
            body.append('\n')
            return
        nodes_to_delete = user_to_last_uses.get(user, [])
        if len(nodes_to_delete):
            to_delete_str = ' = '.join([repr(n) for n in nodes_to_delete] + ['None'])
            body.append(f';  {to_delete_str}\n')
        else:
            body.append('\n')
    prev_stacktrace = None

    def append_stacktrace_summary(node: Node):
        """
            Append a summary of the stacktrace to the generated code. This is
            useful for debugging.
            """
        nonlocal prev_stacktrace
        if node.op not in {'placeholder', 'output'}:
            if node.stack_trace:
                if node.stack_trace != prev_stacktrace:
                    prev_stacktrace = node.stack_trace
                    summary_str = ''
                    parsed_stack_trace = _parse_stack_trace(node.stack_trace)
                    if parsed_stack_trace is not None:
                        lineno = parsed_stack_trace.lineno
                        code = parsed_stack_trace.code
                        summary_str = f'File: {parsed_stack_trace.file}:{lineno}, code: {code}'
                    body.append(f'\n# {summary_str}\n')
            elif prev_stacktrace != '':
                prev_stacktrace = ''
                body.append('\n# No stacktrace found for following nodes\n')

    def stringify_shape(shape: torch.Size) -> str:
        return f'[{', '.join((str(x) for x in shape))}]'

    def emit_node(node: Node):
        maybe_type_annotation = '' if node.type is None else f' : {type_repr(node.type)}'
        if verbose:
            from torch._subclasses.fake_tensor import FakeTensor
            from torch.fx.experimental.proxy_tensor import py_sym_types
            from torch.fx.passes.shape_prop import TensorMetadata
            meta_val = node.meta.get('val', node.meta.get('tensor_meta', None))
            if isinstance(meta_val, FakeTensor):
                maybe_type_annotation = f': "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}"'
            elif isinstance(meta_val, py_sym_types):
                maybe_type_annotation = f': "Sym({meta_val})"'
            elif isinstance(meta_val, TensorMetadata):
                maybe_type_annotation = f': "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}"'
        if node.op == 'placeholder':
            assert isinstance(node.target, str)
            maybe_default_arg = '' if not node.args else f' = {_get_repr(node.args[0])}'
            free_vars.append(f'{node.target}{maybe_type_annotation}{maybe_default_arg}')
            raw_name = node.target.replace('*', '')
            if raw_name != repr(node):
                body.append(f'{repr(node)} = {raw_name}\n')
            return
        elif node.op == 'call_method':
            assert isinstance(node.target, str)
            body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.target)}({_format_args(node.args[1:], node.kwargs)})')
            return
        elif node.op == 'call_function':
            assert callable(node.target)
            if getattr(node.target, '__module__', '') == '_operator' and node.target.__name__ in magic_methods:
                assert isinstance(node.args, tuple)
                body.append(f'{repr(node)}{maybe_type_annotation} = {magic_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))}')
                return
            if getattr(node.target, '__module__', '') == '_operator' and node.target.__name__ in inplace_methods:
                body.append(f'{inplace_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))};  {repr(node)}{maybe_type_annotation} = {_get_repr(node.args[0])}')
                return
            qualified_name = _get_qualified_name(node.target)
            global_name = add_global(qualified_name, node.target)
            if global_name == 'getattr' and isinstance(node.args, tuple) and isinstance(node.args[1], str) and node.args[1].isidentifier() and (len(node.args) == 2):
                body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.args[1])}')
                return
            body.append(f'{repr(node)}{maybe_type_annotation} = {global_name}({_format_args(node.args, node.kwargs)})')
            if node.meta.get('is_wrapped', False):
                wrapped_fns.setdefault(global_name)
            return
        elif node.op == 'call_module':
            assert isinstance(node.target, str)
            body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})')
            return
        elif node.op == 'get_attr':
            assert isinstance(node.target, str)
            body.append(f'{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}')
            return
        elif node.op == 'output':
            if node.type is not None:
                maybe_return_annotation[0] = f' -> {type_repr(node.type)}'
            body.append(self.generate_output(node.args[0]))
            return
        raise NotImplementedError(f'node: {node.op} {node.target}')
    for i, node in enumerate(nodes):
        if verbose:
            append_stacktrace_summary(node)
        body.append(f'# COUNTER: {i}\n')
        emit_node(node)
        delete_unused_values(node)
    if len(body) == 0:
        body.append('pass\n')
    if len(wrapped_fns) > 0:
        wrap_name = add_global('wrap', torch.fx.wrap)
        wrap_stmts = '\n'.join([f'{wrap_name}("{name}")' for name in wrapped_fns])
    else:
        wrap_stmts = ''
    if self._body_transformer:
        body = self._body_transformer(body)
    for name, value in self.additional_globals():
        add_global(name, value)
    prologue = self.gen_fn_def(free_vars, maybe_return_annotation[0])
    lineno_map: Dict[int, Optional[int]] = {}
    prologue_len = prologue.count('\n') + 1
    new_lines: List[str] = []
    cur_idx = None
    for line in ''.join(body).split('\n'):
        counter = re.search('# COUNTER: (\\d+)', line)
        if counter and counter.group(1) is not None:
            cur_idx = int(counter.group(1))
        else:
            lineno_map[len(new_lines) + prologue_len] = cur_idx
            new_lines.append(line)
    code = '\n'.join(new_lines).lstrip('\n')
    code = '\n'.join(('    ' + line for line in code.split('\n')))
    fn_code = f'\n{wrap_stmts}\n\n{prologue}\n{code}'
    return PythonCode(fn_code, globals_, _lineno_map=lineno_map)