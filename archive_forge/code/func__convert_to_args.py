import functools
import inspect
import pennylane as qml
def _convert_to_args(sig, args, kwargs):
    """
    Given the signature of a function, convert the positional and
    keyword arguments to purely positional arguments.
    """
    new_args = []
    for i, param in enumerate(sig):
        if param in kwargs:
            new_args.append(kwargs[param])
        else:
            new_args.append(args[i])
    return tuple(new_args)