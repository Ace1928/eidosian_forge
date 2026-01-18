import collections
def _check_keys_and_fill(name, arg, defaults):

    def _checks(item, defaults):
        if item is None:
            return
        if not isinstance(item, dict):
            raise ValueError("\nElements of the '{name}' argument to make_subplots must be dictionaries or None.\n    Received value of type {typ}: {val}".format(name=name, typ=type(item), val=repr(item)))
        for k in item:
            if k not in defaults:
                raise ValueError("\nInvalid key specified in an element of the '{name}' argument to make_subplots: {k}\n    Valid keys include: {valid_keys}".format(k=repr(k), name=name, valid_keys=repr(list(defaults))))
        for k, v in defaults.items():
            item.setdefault(k, v)
    for arg_i in arg:
        if isinstance(arg_i, (list, tuple)):
            for arg_ii in arg_i:
                _checks(arg_ii, defaults)
        elif isinstance(arg_i, dict):
            _checks(arg_i, defaults)