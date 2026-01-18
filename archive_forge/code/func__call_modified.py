from __future__ import annotations
import functools
import typing
import warnings
def _call_modified(fn: Callable[Concatenate[MonitoredList, ArgSpec], Ret]) -> Callable[Concatenate[MonitoredList, ArgSpec], Ret]:

    @functools.wraps(fn)
    def call_modified_wrapper(self: MonitoredList, *args: ArgSpec.args, **kwargs: ArgSpec.kwargs) -> Ret:
        rval = fn(self, *args, **kwargs)
        self._modified()
        return rval
    return call_modified_wrapper