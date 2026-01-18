from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class IllegalBackReference(ValueError):
    """Too many backrefs exist in a path.

    This error will occur if the back references in a path would be
    outside of the root. For example, ``"/foo/../../"``, contains two back
    references which would reference a directory above the root.

    Note:
        This exception is a subclass of `ValueError` as it is not
        strictly speaking an issue with a filesystem or resource.

    """

    def __init__(self, path):
        self.path = path
        msg = "path '{path}' contains back-references outside of filesystem".format(path=path)
        super(IllegalBackReference, self).__init__(msg)

    def __reduce__(self):
        return (type(self), (self.path,))