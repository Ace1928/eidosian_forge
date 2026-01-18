from typing import Any, Callable, List, Optional, Union
import torch
@torch.no_grad()
def _reattach_grads(self) -> None:
    """
        Given the parameters gradients which have been registered previously, rebuild the whole bucket
        """
    assert len(self._params) > 0
    self._fill = 0
    for p in self._params:
        self._add_grad_as_view(p, keep_existing_value=False)