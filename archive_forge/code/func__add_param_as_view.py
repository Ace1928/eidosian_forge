from typing import Any, Callable, List, Optional, Union
import torch
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