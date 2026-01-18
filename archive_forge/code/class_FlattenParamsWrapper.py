from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
class FlattenParamsWrapper(nn.Module):
    """
    A wrapper for transparently flattening a Module's parameters.

    Compared to the original implementation [1], this version:
    - removes tracing
    - supports shared parameters
    - handles state_dict/load_state_dict transparently
    - is renamed to FlattenParamsWrapper
    - refactored to use the FlatParameter class
    - extended to support flattening multiple groups of params (useful
      when different groups of params need different hyperparameters, like
      learning rate or weight decay)

    [1] https://github.com/SsnL/PyTorch-Reparam-Module

    Args:
        module (nn.Module):
            The module to wrap.
        param_list (Optional[List[List[nn.Parameter]]]):
            Only flatten parameters appearing in the given groups.
            If the param_list is an empty list, then no parameters will get flattened.
            Note, if a single param is in one of the list, it still get flattened and the
            original param is removed and replaced with the flatten one.
            Default: None, flatten all parameters (if any)
        flat_param_names (Optional[List[str]]):
            originally, give each flat_param a unique name. Note a "flat_param_"
            prefix will be added to those names.
    """

    def __init__(self, module: nn.Module, param_list: ParamGroups=None, flat_param_names: Optional[List[str]]=None):
        super().__init__()
        self._fpw_module = module
        self.is_flattened = False
        if param_list is None:
            param_list = list(module.parameters())
        if len(param_list) > 0 and isinstance(param_list[0], nn.Parameter):
            param_list = [cast(List[nn.Parameter], param_list)]
        self.num_params_managed = 0
        self._param_sets = []
        overall_param_set: Set[nn.Parameter] = set()
        for p_list in param_list:
            p_set: Set[nn.Parameter] = set(cast(List[nn.Parameter], p_list))
            self.num_params_managed += len(p_set)
            overall_param_set = overall_param_set.union(p_set)
            new_p_set_with_names = set()
            for m in self.modules():
                for n, p in m.named_parameters(recurse=False):
                    if p in p_set:
                        new_p_set_with_names.add((m, n))
            if new_p_set_with_names:
                self._param_sets.append(new_p_set_with_names)
        if len(overall_param_set) != self.num_params_managed:
            raise ValueError(f'Incorrect param groups {len(overall_param_set)} vs {self.num_param_managed}')
        self.flat_params: List[nn.Parameter] = []
        if flat_param_names is None:
            flat_param_names = [f'{i}' for i, _ in enumerate(self._param_sets)]
        if len(flat_param_names) != len(self._param_sets):
            raise ValueError('Names and number of param lists must be equal')
        if len(flat_param_names) != len(set(flat_param_names)):
            raise ValueError('Each flat param must be given a unique name')
        self.flat_param_names = [f'flat_param_{n}' for n in flat_param_names]
        flat_param: Optional[nn.Parameter] = None
        for new_p_set in self._param_sets:
            params, param_infos, shared_param_infos = self._init_flatten_params(new_p_set)
            flat_param = FlatParameter(params, params[0].requires_grad)
            flat_param._param_infos = param_infos
            flat_param._shared_param_infos = shared_param_infos
            self.flat_params.append(flat_param)
        self._flatten_params(self.flat_params)
        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(_pre_load_state_dict_hook)
        self._auto_unflatten_state_dict = True

    @property
    def module(self) -> Any:
        """Support fpw.module in case we are immitating DDP, which has .module
        property to the underlying module.
        """
        return self._fpw_module

    @property
    def flat_param(self) -> nn.Parameter:
        """We used to support only a single flat_param. This allows us to
        be backward compatible.
        """
        assert len(self.flat_params) == 1, f'Incorrect access to flat_param: len(self.flat_params)={len(self.flat_params)}'
        return self.flat_params[0]

    def _init_flatten_params(self, p_set: Set[Tuple[nn.Module, str]]) -> Tuple[List[nn.Parameter], List[Tuple[str, nn.Module, str]], List[Tuple[str, str, nn.Module, str, nn.Module, str]]]:
        """Build metadata for need-to-be-flatten parameters and returns a list
            contains the need-to-be-flatten parameters.

            This also returns param_infos and shared_param_infos, which
            will be attached to the flat parameter object.

        Args:
            p_set (set):
                A set of (module, param_name) for a set of params that needed
                to be flattened. There could be shared params in this set.
        """
        param_infos = []
        shared_param_memo: Dict[nn.Parameter, Tuple[str, nn.Module, str]] = {}
        shared_param_infos = []
        params = []
        fp32 = []
        fp16 = []
        for module_name, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p.dtype != torch.float16:
                    fp32.append(module_name)
                else:
                    fp16.append(module_name)
                if p is not None and (m, n) in p_set:
                    if p in shared_param_memo:
                        mname, shared_m, shared_n = shared_param_memo[p]
                        shared_param_infos.append((module_name, mname, m, n, shared_m, shared_n))
                    else:
                        shared_param_memo[p] = (module_name, m, n)
                        param_infos.append((module_name, m, n))
                        params.append(p)
        del shared_param_memo
        fp16_msg, fp32_msg = (','.join(fp16), ','.join(fp32))
        assert len(set((p.dtype for p in params))) == 1, f'expects all parameters to have same dtype: fp32: {fp32_msg} \n fp16: {fp16_msg} '
        assert len(set((p.requires_grad for p in params))) == 1, f'expects all parameters to have same requires_grad {p_set}'
        assert len(params) == len(set(params)), 'params list should not have dups'
        return (params, param_infos, shared_param_infos)

    @property
    def _param_infos(self) -> Iterator[Tuple[str, nn.Module, str]]:
        return chain(*[p._param_infos for p in self.flat_params])

    @property
    def _shared_param_infos(self) -> Iterator[Tuple[str, str, nn.Module, str, nn.Module, str]]:
        return chain(*[p._shared_param_infos for p in self.flat_params])

    def _flatten_params(self, flat_params: List[nn.Parameter]) -> None:
        """Flatten the managed parameters and replaced the original
        attributes with views to the flat params.
        """
        assert not self.is_flattened
        self.is_flattened = True
        assert len(self.flat_param_names) == len(flat_params), f'{len(self.flat_param_names)} vs. {len(flat_params)}'
        for n, flat_param in zip(self.flat_param_names, flat_params):
            self.register_parameter(n, flat_param)
        self.flat_params = flat_params
        for _, m, n in self._param_infos:
            delattr(m, n)
        for _, _, m, n, _, _ in self._shared_param_infos:
            delattr(m, n)
        self._unflatten_params_as_views()

    def _unflatten_params(self, external_data: Optional[List[Optional[Tensor]]]=None) -> None:
        """Undo flattening and create separate parameters from the already flattened
        self.flat_param or a user supplied external data.
        """
        assert self.is_flattened or external_data is not None
        self.is_flattened = False
        ps = self.get_param_views(external_data)
        for (_, m, n), p in zip(self._param_infos, ps):
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, nn.Parameter(p))
        for _, _, m, n, shared_m, shared_n in self._shared_param_infos:
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, getattr(shared_m, shared_n))
        if hasattr(self._fpw_module, '_unflattened_param_views'):
            delattr(self._fpw_module, '_unflattened_param_views')
        for n in self.flat_param_names:
            delattr(self, n)
        self.flat_params = []

    def _unflatten_params_as_views(self) -> None:
        """Unlike ``_unflatten_params``, this function unflatten into views and keep
        self.flat_param unchanged.
        """
        assert self.is_flattened
        ps = self.get_param_views()
        param_views = []
        for (_, m, n), p in zip(self._param_infos, ps):
            setattr(m, n, p)
            param_views.append(p)
        setattr(self._fpw_module, '_unflattened_param_views', param_views)
        for _, _, m, n, shared_m, shared_n in self._shared_param_infos:
            setattr(m, n, getattr(shared_m, shared_n))

    @contextmanager
    def unflatten_params(self, flat_params: Optional[List[Tensor]]=None) -> Generator:
        """
        Unflatten params. If the current instance is already unflattened, then
        it will remain unflattened after the context manager exits.

        Args:
            flat_params (List[Tensor], Optional):
                flat params to use for unflattening.
                If provided, the current instance must be in a flattened state
                at the start of the context manager. The provided Tensor must be
                appropriately sized and will only be used within the context
                manager. After the context manager exits, we will revert to
                using ``self.flat_params``
                Default: None.
        """
        assert flat_params is None or self.is_flattened, 'Unflattening with external flat_param requires current instance to be flattened'
        orig_flattened = self.is_flattened
        if orig_flattened:
            orig_flat_params = self.flat_params
            self._unflatten_params(cast(Optional[List[Optional[Tensor]]], flat_params))
        try:
            yield
        finally:
            if orig_flattened:
                self._flatten_params(orig_flat_params)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.module.__getitem__(key)

    @typing.overload
    def state_dict(self, destination: Mapping[str, Tensor], prefix: str=..., keep_vars: bool=...) -> Mapping[str, Tensor]:
        ...

    @typing.overload
    def state_dict(self, prefix: str=..., keep_vars: bool=...) -> 'OrderedDict[str, Tensor]':
        ...

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return the wrapped module's state_dict."""
        if self.is_flattened and self._auto_unflatten_state_dict:
            with self.unflatten_params():
                return super().state_dict(*args, **kwargs)
        else:
            return super().state_dict(*args, **kwargs)

    def flat_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return the flattened state_dict."""
        assert self.is_flattened
        with self._no_auto_unflatten_state_dict():
            return self.state_dict(*args, **kwargs)

    @contextmanager
    def _no_auto_unflatten_state_dict(self) -> Generator:
        backup = self._auto_unflatten_state_dict
        self._auto_unflatten_state_dict = False
        try:
            yield
        finally:
            self._auto_unflatten_state_dict = backup

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], 'OrderedDict[str, Tensor]'], strict: bool=True) -> NamedTuple:
        """
        Load a state dict. If necessary, ``unflatten_params`` will be called to
        match the input state_dict.
        """
        if self.num_params_managed > 0 and self.is_flattened and (not any((k.startswith('flat_param_') for k in state_dict.keys()))):
            with self.unflatten_params():
                return super().load_state_dict(state_dict, strict)
        else:
            if 'flat_param' in state_dict:
                state_dict['flat_param_0'] = state_dict['flat_param']
                del state_dict['flat_param']
            return super().load_state_dict(state_dict, strict)

    def forward(self, *inputs: Any, **kwinputs: Any) -> Any:
        self._unflatten_params_as_views()
        return self.module(*inputs, **kwinputs)

    def get_param_views(self, external_data_list: Optional[List[Optional[Tensor]]]=None) -> Iterator[Tensor]:
        """Used to get a generator over all views from a list of external data list."""
        params = self.flat_params
        if external_data_list is None:
            external_data_list = [None] * len(params)
        assert len(external_data_list) == len(params), f'Incorrect external data list: {len(external_data_list)} vs. {len(params)}'
        gens = []
        for p, data in zip(params, external_data_list):
            gens.append(p.get_param_views(data))
        return chain(*gens)

    def metadata(self, flat_param_idx: int) -> Tuple[List[str], Sequence[torch.Size], List[int]]:
        """Return metadata for a flat param given its index in the flat_params list."""
        return self.flat_params[flat_param_idx].metadata()