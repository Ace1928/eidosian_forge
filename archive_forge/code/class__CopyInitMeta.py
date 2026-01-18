import typing
import abc
import six
from appdirs import AppDirs
from ._repr import make_repr
from .osfs import OSFS
class _CopyInitMeta(abc.ABCMeta):
    """A metaclass that performs a hard copy of the `__init__`.

    This is a fix for Sphinx, which is a pain to configure in a way that
    it documents the ``__init__`` method of a class when it is inherited.
    Copying ``__init__`` makes it think it is not inherited, and let us
    share the documentation between all the `_AppFS` subclasses.

    """

    def __new__(mcls, classname, bases, cls_dict):
        cls_dict.setdefault('__init__', bases[0].__init__)
        return super(abc.ABCMeta, mcls).__new__(mcls, classname, bases, cls_dict)