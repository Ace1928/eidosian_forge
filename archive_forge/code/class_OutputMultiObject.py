from collections.abc import Sequence
from traits import __version__ as traits_version
import traits.api as traits
from traits.api import TraitType, Unicode
from traits.trait_base import _Undefined
from pathlib import Path
from ...utils.filemanip import path_resolve
class OutputMultiObject(MultiObject):
    """Implements a user friendly traits that accepts one or more
    paths to files or directories. This is the output version which
    return a single string whenever possible (when it was set to a
    single value or a list of length 1). Default value of this trait
    is _Undefined. It does not accept empty lists.

    XXX This should only be used as a final resort. We should stick to
    established Traits to the extent possible.

    XXX This needs to be vetted by somebody who understands traits

    >>> from nipype.interfaces.base import OutputMultiObject, TraitedSpec
    >>> class A(TraitedSpec):
    ...     foo = OutputMultiObject(File(exists=False))
    >>> a = A()
    >>> a.foo
    <undefined>

    >>> a.foo = '/software/temp/foo.txt'
    >>> a.foo
    '/software/temp/foo.txt'

    >>> a.foo = ['/software/temp/foo.txt']
    >>> a.foo
    '/software/temp/foo.txt'

    >>> a.foo = ['/software/temp/foo.txt', '/software/temp/goo.txt']
    >>> a.foo
    ['/software/temp/foo.txt', '/software/temp/goo.txt']

    """

    def get(self, objekt, name):
        value = self.get_value(objekt, name)
        if len(value) == 0:
            return Undefined
        elif len(value) == 1:
            return value[0]
        else:
            return value

    def set(self, objekt, name, value):
        self.set_value(objekt, name, value)