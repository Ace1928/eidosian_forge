import sys as _sys
from keyword import iskeyword as _iskeyword
def _make_tuple_bunch(typename, field_names, extra_field_names=None, module=None):
    """
    Create a namedtuple-like class with additional attributes.

    This function creates a subclass of tuple that acts like a namedtuple
    and that has additional attributes.

    The additional attributes are listed in `extra_field_names`.  The
    values assigned to these attributes are not part of the tuple.

    The reason this function exists is to allow functions in SciPy
    that currently return a tuple or a namedtuple to returned objects
    that have additional attributes, while maintaining backwards
    compatibility.

    This should only be used to enhance *existing* functions in SciPy.
    New functions are free to create objects as return values without
    having to maintain backwards compatibility with an old tuple or
    namedtuple return value.

    Parameters
    ----------
    typename : str
        The name of the type.
    field_names : list of str
        List of names of the values to be stored in the tuple. These names
        will also be attributes of instances, so the values in the tuple
        can be accessed by indexing or as attributes.  At least one name
        is required.  See the Notes for additional restrictions.
    extra_field_names : list of str, optional
        List of names of values that will be stored as attributes of the
        object.  See the notes for additional restrictions.

    Returns
    -------
    cls : type
        The new class.

    Notes
    -----
    There are restrictions on the names that may be used in `field_names`
    and `extra_field_names`:

    * The names must be unique--no duplicates allowed.
    * The names must be valid Python identifiers, and must not begin with
      an underscore.
    * The names must not be Python keywords (e.g. 'def', 'and', etc., are
      not allowed).

    Examples
    --------
    >>> from scipy._lib._bunch import _make_tuple_bunch

    Create a class that acts like a namedtuple with length 2 (with field
    names `x` and `y`) that will also have the attributes `w` and `beta`:

    >>> Result = _make_tuple_bunch('Result', ['x', 'y'], ['w', 'beta'])

    `Result` is the new class.  We call it with keyword arguments to create
    a new instance with given values.

    >>> result1 = Result(x=1, y=2, w=99, beta=0.5)
    >>> result1
    Result(x=1, y=2, w=99, beta=0.5)

    `result1` acts like a tuple of length 2:

    >>> len(result1)
    2
    >>> result1[:]
    (1, 2)

    The values assigned when the instance was created are available as
    attributes:

    >>> result1.y
    2
    >>> result1.beta
    0.5
    """
    if len(field_names) == 0:
        raise ValueError('field_names must contain at least one name')
    if extra_field_names is None:
        extra_field_names = []
    _validate_names(typename, field_names, extra_field_names)
    typename = _sys.intern(str(typename))
    field_names = tuple(map(_sys.intern, field_names))
    extra_field_names = tuple(map(_sys.intern, extra_field_names))
    all_names = field_names + extra_field_names
    arg_list = ', '.join(field_names)
    full_list = ', '.join(all_names)
    repr_fmt = ''.join(('(', ', '.join((f'{name}=%({name})r' for name in all_names)), ')'))
    tuple_new = tuple.__new__
    _dict, _tuple, _zip = (dict, tuple, zip)
    s = f"""def __new__(_cls, {arg_list}, **extra_fields):\n    return _tuple_new(_cls, ({arg_list},))\n\ndef __init__(self, {arg_list}, **extra_fields):\n    for key in self._extra_fields:\n        if key not in extra_fields:\n            raise TypeError("missing keyword argument '%s'" % (key,))\n    for key, val in extra_fields.items():\n        if key not in self._extra_fields:\n            raise TypeError("unexpected keyword argument '%s'" % (key,))\n        self.__dict__[key] = val\n\ndef __setattr__(self, key, val):\n    if key in {repr(field_names)}:\n        raise AttributeError("can't set attribute %r of class %r"\n                             % (key, self.__class__.__name__))\n    else:\n        self.__dict__[key] = val\n"""
    del arg_list
    namespace = {'_tuple_new': tuple_new, '__builtins__': dict(TypeError=TypeError, AttributeError=AttributeError), '__name__': f'namedtuple_{typename}'}
    exec(s, namespace)
    __new__ = namespace['__new__']
    __new__.__doc__ = f'Create new instance of {typename}({full_list})'
    __init__ = namespace['__init__']
    __init__.__doc__ = f'Instantiate instance of {typename}({full_list})'
    __setattr__ = namespace['__setattr__']

    def __repr__(self):
        """Return a nicely formatted representation string"""
        return self.__class__.__name__ + repr_fmt % self._asdict()

    def _asdict(self):
        """Return a new dict which maps field names to their values."""
        out = _dict(_zip(self._fields, self))
        out.update(self.__dict__)
        return out

    def __getnewargs_ex__(self):
        """Return self as a plain tuple.  Used by copy and pickle."""
        return (_tuple(self), self.__dict__)
    for method in (__new__, __repr__, _asdict, __getnewargs_ex__):
        method.__qualname__ = f'{typename}.{method.__name__}'
    class_namespace = {'__doc__': f'{typename}({full_list})', '_fields': field_names, '__new__': __new__, '__init__': __init__, '__repr__': __repr__, '__setattr__': __setattr__, '_asdict': _asdict, '_extra_fields': extra_field_names, '__getnewargs_ex__': __getnewargs_ex__}
    for index, name in enumerate(field_names):

        def _get(self, index=index):
            return self[index]
        class_namespace[name] = property(_get)
    for name in extra_field_names:

        def _get(self, name=name):
            return self.__dict__[name]
        class_namespace[name] = property(_get)
    result = type(typename, (tuple,), class_namespace)
    if module is None:
        try:
            module = _sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass
    if module is not None:
        result.__module__ = module
        __new__.__module__ = module
    return result