from __future__ import (absolute_import, division, print_function)
import itertools
import os
import os.path
import pkgutil
import re
import sys
from keyword import iskeyword
from tokenize import Name as _VALID_IDENTIFIER_REGEX
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.six import string_types, PY3
from ._collection_config import AnsibleCollectionConfig
from contextlib import contextmanager
from types import ModuleType
class _AnsibleNSTraversable:
    """Class that implements the ``importlib.resources.abc.Traversable``
    interface for the following ``ansible_collections`` namespace packages::

    * ``ansible_collections``
    * ``ansible_collections.<namespace>``

    These namespace packages operate differently from a normal Python
    namespace package, in that the same namespace can be distributed across
    multiple directories on the filesystem and still function as a single
    namespace, such as::

    * ``/usr/share/ansible/collections/ansible_collections/ansible/posix/``
    * ``/home/user/.ansible/collections/ansible_collections/ansible/windows/``

    This class will mimic the behavior of various ``pathlib.Path`` methods,
    by combining the results of multiple root paths into the output.

    This class does not do anything to remove duplicate collections from the
    list, so when traversing either namespace patterns supported by this class,
    it is possible to have the same collection located in multiple root paths,
    but precedence rules only use one. When iterating or traversing these
    package roots, there is the potential to see the same collection in
    multiple places without indication of which would be used. In such a
    circumstance, it is best to then call ``importlib.resources.files`` for an
    individual collection package rather than continuing to traverse from the
    namespace package.

    Several methods will raise ``NotImplementedError`` as they do not make
    sense for these namespace packages.
    """

    def __init__(self, *paths):
        self._paths = [pathlib.Path(p) for p in paths]

    def __repr__(self):
        return "_AnsibleNSTraversable('%s')" % "', '".join(map(to_text, self._paths))

    def iterdir(self):
        return itertools.chain.from_iterable((p.iterdir() for p in self._paths if p.is_dir()))

    def is_dir(self):
        return any((p.is_dir() for p in self._paths))

    def is_file(self):
        return False

    def glob(self, pattern):
        return itertools.chain.from_iterable((p.glob(pattern) for p in self._paths if p.is_dir()))

    def _not_implemented(self, *args, **kwargs):
        raise NotImplementedError('not usable on namespaces')
    joinpath = __truediv__ = read_bytes = read_text = _not_implemented