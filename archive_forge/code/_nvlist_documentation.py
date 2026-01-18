import numbers
from collections import namedtuple
from contextlib import contextmanager
from .bindings import libnvpair
from .ctypes import _type_to_suffix

    A context manager that allocates a pointer to a C nvlist_t and yields
    a CData object representing a pointer to the pointer via 'as' target.
    The caller can pass that pointer to a pointer to a C function that
    creates a new nvlist_t object.
    The context manager takes care of memory management for the nvlist_t
    and also populates the 'props' dictionary with data from the nvlist_t
    upon leaving the 'with' block.

    :param dict props: the dictionary to be populated with data from the nvlist.
    :return: an FFI CData object representing the pointer to nvlist_t pointer.
    :rtype: CData
    