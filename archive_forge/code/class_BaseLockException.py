import typing
class BaseLockException(Exception):
    LOCK_FAILED = 1

    def __init__(self, *args: typing.Any, fh: typing.Union[typing.IO, None, int]=None, **kwargs: typing.Any) -> None:
        self.fh = fh
        Exception.__init__(self, *args)