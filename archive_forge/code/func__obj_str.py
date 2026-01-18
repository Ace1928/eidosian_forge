import typing
def _obj_str(obj: typing.Any, default: str) -> str:
    to_str_funcs: typing.List[typing.Callable] = [str, repr]
    for func in to_str_funcs:
        try:
            obj = func(obj)
        except (UnicodeError, TypeError):
            continue
        else:
            return obj
    else:
        return default