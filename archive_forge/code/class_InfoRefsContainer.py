import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
class InfoRefsContainer(RefsContainer):
    """Refs container that reads refs from a info/refs file."""

    def __init__(self, f) -> None:
        self._refs = {}
        self._peeled = {}
        for line in f.readlines():
            sha, name = line.rstrip(b'\n').split(b'\t')
            if name.endswith(PEELED_TAG_SUFFIX):
                name = name[:-3]
                if not check_ref_format(name):
                    raise ValueError('invalid ref name %r' % name)
                self._peeled[name] = sha
            else:
                if not check_ref_format(name):
                    raise ValueError('invalid ref name %r' % name)
                self._refs[name] = sha

    def allkeys(self):
        return self._refs.keys()

    def read_loose_ref(self, name):
        return self._refs.get(name, None)

    def get_packed_refs(self):
        return {}

    def get_peeled(self, name):
        try:
            return self._peeled[name]
        except KeyError:
            return self._refs[name]