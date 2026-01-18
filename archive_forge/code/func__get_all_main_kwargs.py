import copy
import logging
from s3transfer.utils import get_callbacks
def _get_all_main_kwargs(self):
    kwargs = copy.copy(self._main_kwargs)
    for key, pending_value in self._pending_main_kwargs.items():
        if isinstance(pending_value, list):
            result = []
            for future in pending_value:
                result.append(future.result())
        else:
            result = pending_value.result()
        kwargs[key] = result
    return kwargs