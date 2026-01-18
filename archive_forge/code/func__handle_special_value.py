from abc import ABCMeta
@staticmethod
def _handle_special_value(value):
    if isinstance(value, classmethod) or isinstance(value, staticmethod):
        value = value.__get__(None, dict)
    elif isinstance(value, property):
        value = value.fget
    return value