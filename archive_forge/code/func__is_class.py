import base64
import inspect
import builtins
@classmethod
def _is_class(cls, dict_):
    assert isinstance(dict_, dict)
    if len(dict_) != 1:
        return False
    k = list(dict_.keys())[0]
    if not isinstance(k, (bytes, str)):
        return False
    for p in cls._class_prefixes:
        if k.startswith(p):
            return True
    for p in cls._class_suffixes:
        if k.endswith(p):
            return True
    return False