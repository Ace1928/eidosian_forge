import typing
def _get_fullqual_name(func: typing.Callable) -> str:
    return f'{func.__module__}.{func.__qualname__}'