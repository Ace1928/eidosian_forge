import functools
import operator
def _flatten_value(obj, key_path, strict=False):
    return [('.'.join(key_path), _canonicalize(obj, strict=strict))]