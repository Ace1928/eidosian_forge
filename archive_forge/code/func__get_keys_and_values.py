import re
def _get_keys_and_values(structure, key_spec):
    if callable(key_spec):
        arity = _get_arity(key_spec)
        if arity == 1:
            return [(k, v) for k, v in _items(structure) if key_spec(k)]
        elif arity == 2:
            return [(k, v) for k, v in _items(structure) if key_spec(k, v)]
        else:
            raise ValueError('callable in transform path must take 1 or 2 arguments')
    return [(key_spec, _get(structure, key_spec, _EMPTY_SENTINEL))]