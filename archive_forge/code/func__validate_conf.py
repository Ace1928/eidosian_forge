from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def _validate_conf(self) -> None:
    """Validating if the config is valid.

        This includes asserting the following:
        1. validating that one and only one of top_k_element and top_k_percent is set.
        2. Asserting that both element and percentage are in valid ranges.

        Throws:
            ValueError:
                If validation fails.
        """

    def both_set(a: Optional[int], b: Optional[float]) -> bool:
        return a is not None and b is not None
    if both_set(self._sst_top_k_element, self._sst_top_k_percent) or both_set(self._dst_top_k_element, self._dst_top_k_percent):
        raise ValueError(f"top_k_element and top_k_percent can't be both set\nInput values are: sst element={self._sst_top_k_element}, sst percent={self._sst_top_k_percent}, dst element={self._dst_top_k_element}, dst percent={self._dst_top_k_percent}")

    def none_or_in_range(a: Optional[float]) -> bool:
        return a is None or 0.0 < a <= 100.0
    if not (none_or_in_range(self._sst_top_k_percent) and none_or_in_range(self._dst_top_k_percent)):
        raise ValueError(f'top_k_percent values for sst and dst has to be in the interval (0, 100].\nInput values are: sst percent={self._sst_top_k_percent}, dst percent={self._dst_top_k_percent}')

    def none_or_greater_0(a: Optional[int]) -> bool:
        return a is None or 0 < a
    if not (none_or_greater_0(self._sst_top_k_element) and none_or_greater_0(self._dst_top_k_element)):
        raise ValueError(f'top_k_element values for sst and dst has to be greater than 0.\nInput values are: sst element={self._sst_top_k_element} and dst element={self._dst_top_k_element}')