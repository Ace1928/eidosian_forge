from contextlib import suppress
from io import TextIOWrapper
from . import abc
def _io_wrapper(file, mode='r', *args, **kwargs):
    if mode == 'r':
        return TextIOWrapper(file, *args, **kwargs)
    elif mode == 'rb':
        return file
    raise ValueError("Invalid mode value '{}', only 'r' and 'rb' are supported".format(mode))