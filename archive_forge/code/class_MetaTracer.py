import torch
import torch.fx
import warnings
import functools
import builtins
from typing import Any, Callable, Dict, Optional, Union
class MetaTracer(torch.fx.Tracer):
    allow_insert_stateless_mods: bool = True
    _TORCH_METHODS_TO_PATCH = ['arange', 'zeros', 'ones', 'full_like', 'eye']

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        rv = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
        if kind == 'placeholder' and target in self.meta_args:
            rv.install_tensor_meta(self.meta_args[target])
            return rv
        if target in self.orig_fns:
            if 'device' in kwargs:
                kwargs['device'] = 'meta'
        try:
            args_metas = torch.fx.node.map_aggregate(args, proxys_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, proxys_to_metas)
            if kind == 'call_function':
                meta_target = manual_meta_overrides.get(target, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == 'call_method':
                meta_out = getattr(args_metas[0], target)(*args_metas[1:], **kwargs_metas)
            elif kind == 'call_module':
                assert hasattr(self, 'orig_forward')
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in manual_meta_overrides:
                        meta_out = manual_meta_overrides[mod_type](mod, *args_metas, **kwargs_metas)
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    self._disable_module_getattr = False
            elif kind == 'get_attr':
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split('.')
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    assert isinstance(attr_itr, torch.Tensor)
                    meta_out = attr_itr.to(device='meta')
                finally:
                    self._disable_module_getattr = False
            else:
                return rv
            assert isinstance(rv, torch.fx.Proxy), 'Dont support composite output yet'
            rv.install_tensor_meta(meta_out)
        except Exception as e:
            warnings.warn(f'Could not compute metadata for {kind} target {target}: {e}')
        return rv

    def getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, '_disable_module_getattr', False):
            return attr_val
        else:
            return super().getattr(attr, attr_val, parameter_proxy_cache)

    def call_module(self, m, forward, args, kwargs):
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    def _insert_module_as_submodule(self, mod: torch.nn.Module) -> str:
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f'{mod_name}_{idx}'
        while hasattr(self.root, path):
            path = f'{mod_name}_{idx}'
            idx += 1
        self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: torch.nn.Module) -> str:
        try:
            return super().path_of_module(mod)
        except NameError as e:
            if self.allow_insert_stateless_mods and len(list(mod.parameters())) == 0 and (len(list(mod.buffers())) == 0):
                path = self._insert_module_as_submodule(mod)
                self.prev_module = path
                return path
            raise

    def proxy(self, node):
        return MetaProxy(node, self)

    def trace(self, root, meta_args: Dict[str, torch.Tensor], concrete_args=None):
        assert isinstance(meta_args, dict)
        self.meta_args = meta_args
        self.patched_torch_methods = {target: gen_constructor_wrapper(getattr(torch, target)) for target in self._TORCH_METHODS_TO_PATCH}
        self.orig_fns = set()
        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)
        try:
            graph = super().trace(root, concrete_args)
            graph._tracer_extras = {'meta_args': meta_args}
            return graph
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)