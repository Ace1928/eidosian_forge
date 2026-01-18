from typing import Optional, Type, IO
from ._typing_compat import Literal
from types import TracebackType
def FileLock(fileobj: IO, mode: int=0, filename: Optional[str]=None) -> BaseLock:
    if has_fcntl:
        return UnixFileLock(fileobj, mode)
    elif has_msvcrt and filename is not None:
        return WindowsFileLock(filename)
    return BaseLock()