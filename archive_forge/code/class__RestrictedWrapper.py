import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
class _RestrictedWrapper(metaclass=_ClassInitMeta):
    """Base class to wrap a Deb822 object, restricting write access to some keys.

    The underlying data is hidden internally.  Subclasses may keep a reference
    to the data before giving it to this class's constructor, if necessary, but
    RestrictedField should cover most use-cases.  The dump method from
    Deb822 is directly proxied.

    Typical usage::

        class Foo(object):
            def __init__(self, ...):
                # ...

            @staticmethod
            def from_str(self, s):
                # Parse s...
                return Foo(...)

            def to_str(self):
                # Return in string format.
                return ...

        class MyClass(deb822._RestrictedWrapper):
            def __init__(self):
                data = Deb822ParagraphElement.new_empty_paragraph()
                data['Bar'] = 'baz'
                super(MyClass, self).__init__(data)

            foo = deb822.RestrictedField(
                    'Foo', from_str=Foo.from_str, to_str=Foo.to_str)

            bar = deb822.RestrictedField('Bar', allow_none=False)

        d = MyClass()
        d['Bar'] # returns 'baz'
        d['Bar'] = 'quux' # raises RestrictedFieldError
        d.bar = 'quux'
        d.bar # returns 'quux'
        d['Bar'] # returns 'quux'

        d.foo = Foo(...)
        d['Foo'] # returns string representation of foo
    """
    __restricted_fields = frozenset()

    @classmethod
    def _class_init(cls, new_attrs):
        restricted_fields = []
        for attr_name, val in new_attrs.items():
            if isinstance(val, RestrictedField):
                restricted_fields.append(val.name.lower())
                cls.__init_restricted_field(attr_name, val)
        cls.__restricted_fields = frozenset(restricted_fields)

    @classmethod
    def __init_restricted_field(cls, attr_name, field):

        def getter(self):
            val = self.__data.get(field.name)
            if field.from_str is not None:
                return field.from_str(val)
            return val

        def setter(self, val):
            if val is not None and field.to_str is not None:
                val = field.to_str(val)
            if val is None:
                if field.allow_none:
                    if field.name in self.__data:
                        del self.__data[field.name]
                else:
                    raise TypeError('value must not be None')
            else:
                self.__data[field.name] = val
        setattr(cls, attr_name, property(getter, setter, None, field.name))

    def __init__(self, data, _internal_validate=True):
        """Initializes the wrapper over 'data', a Deb822ParagraphElement object."""
        super(_RestrictedWrapper, self).__init__()
        if _internal_validate and (not isinstance(data, Deb822NoDuplicateFieldsParagraphElement)):
            raise ValueError('Paragraph has duplicated fields: ' + str(data.__class__.__qualname__))
        self.__data = data

    @property
    def _underlying_paragraph(self):
        return self.__data

    def __getitem__(self, key):
        return self.__data[key]

    def __setitem__(self, key, value):
        if key.lower() in self.__restricted_fields:
            raise RestrictedFieldError('%s may not be modified directly; use the associated property' % key)
        self.__data[key] = value

    def __delitem__(self, key):
        if key.lower() in self.__restricted_fields:
            raise RestrictedFieldError('%s may not be modified directly; use the associated property' % key)
        del self.__data[key]

    def __iter__(self):
        return (str(k) for k in self.__data)

    def __len__(self):
        return len(self.__data)

    def dump(self, fd=None, encoding=None, text_mode=False):
        """Calls dump() on the underlying data object.

        See Deb822.dump for more information.
        """
        if fd is not None:
            if encoding is None and (not text_mode):
                self.__data.dump(cast('IO[bytes]', fd))
                return None
            as_str = self.__data.dump()
            if encoding is not None:
                cast('IO[bytes]', fd).write(as_str.encode(encoding))
            elif text_mode:
                cast('IO[str]', fd).write(as_str)
            return None
        return self.__data.dump()