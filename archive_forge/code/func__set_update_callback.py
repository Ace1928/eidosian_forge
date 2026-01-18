import abc
import typing as t
from .interface.summary_record import SummaryItem, SummaryRecord
def _set_update_callback(self, update_callback: t.Callable):
    object.__setattr__(self, '_update_callback', update_callback)