from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import six
class DataObject(six.with_metaclass(_DataType, tuple)):
    """Parent class of dumb data object."""

    def __new__(cls, **kwargs):
        names = getattr(cls, 'NAMES', tuple())
        invalid_names = set(kwargs) - set(names)
        if invalid_names:
            raise ValueError('Invalid names: ' + repr(invalid_names))
        tpl = tuple((kwargs[name] if name in kwargs else None for name in names))
        return super(DataObject, cls).__new__(cls, tpl)

    def replace(self, **changes):
        out = dict(((n, changes.get(n, getattr(self, n, None))) for n in self.NAMES))
        return self.__class__(**out)