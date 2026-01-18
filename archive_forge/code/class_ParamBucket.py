from typing import Any, Callable, List, Optional, Union
import torch
class ParamBucket(Bucket):
    """
    Helper class to simplify the handling of parameter buckets
    """

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(size, dtype, device)

    def to(self, device: Optional[Union[int, torch.device]], dtype: Optional[torch.dtype]=None, non_blocking: bool=False, keep_param_alignment: bool=True) -> 'ParamBucket':
        """
        Move the underlying buffer
        """
        super().to(device, dtype, non_blocking)
        if keep_param_alignment:
            self._reattach_params()

    @torch.no_grad()
    def add_param(self, param: torch.Tensor) -> None:
        """
        Add a new parameter gradient to the bucket. Param.grad becomes a view of this bucket buffer
        """
        assert id(param) not in self._param_ids, 'The same param cannot be checked in twice'
        self._add_param_as_view(param)
        self._params.append(param)
        self._param_ids.append(id(param))

    @torch.no_grad()
    def _add_param_as_view(self, param: torch.Tensor, keep_existing_value: bool=True) -> None:
        assert self.buffer is not None
        assert param.dtype == self.buffer.dtype, f'Different types for the bucket and the param, cannot proceed: {param.dtype} - {self.buffer.dtype}'
        assert param.device == self.buffer.device, f'Different devices for the bucket and the param, cannot proceed: {param.device} - {self.buffer.device}'
        fill_next = self._fill + param.numel()
        assert fill_next <= self.buffer.numel()
        if keep_existing_value:
            self.buffer[self._fill:fill_next].copy_(param.data.flatten())
        param.data = self.buffer[self._fill:fill_next].view_as(param.data)
        self._fill = fill_next

    @torch.no_grad()
    def _reattach_params(self) -> None:
        """
        Given the parameters which have been registered previously, rebuild the whole bucket
        """
        assert len(self._params) > 0
        self._fill = 0
        for p in self._params:
            if p.dtype != self.buffer.dtype:
                p.data = p.data.to(self.buffer.dtype)
            self._add_param_as_view(p, keep_existing_value=False)