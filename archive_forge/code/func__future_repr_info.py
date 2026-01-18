import reprlib
from _thread import get_ident
from . import format_helpers
def _future_repr_info(future):
    """helper function for Future.__repr__"""
    info = [future._state.lower()]
    if future._state == _FINISHED:
        if future._exception is not None:
            info.append(f'exception={future._exception!r}')
        else:
            result = reprlib.repr(future._result)
            info.append(f'result={result}')
    if future._callbacks:
        info.append(_format_callbacks(future._callbacks))
    if future._source_traceback:
        frame = future._source_traceback[-1]
        info.append(f'created at {frame[0]}:{frame[1]}')
    return info