from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import six
class _DataType(type):
    """Dumb immutable data type."""

    def __new__(cls, classname, bases, class_dict):
        class_dict = class_dict.copy()
        names = class_dict.get('NAMES', tuple())
        class_dict.update(((name, cls._CreateAccessor(i)) for i, name in enumerate(names)))
        return super(_DataType, cls).__new__(cls, classname, bases, class_dict)

    @staticmethod
    def _CreateAccessor(i):
        """Create an tuple accessor property."""
        return property(lambda tpl: tpl[i])