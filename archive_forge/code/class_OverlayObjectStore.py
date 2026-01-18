import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
class OverlayObjectStore(BaseObjectStore):
    """Object store that can overlay multiple object stores."""

    def __init__(self, bases, add_store=None) -> None:
        self.bases = bases
        self.add_store = add_store

    def add_object(self, object):
        if self.add_store is None:
            raise NotImplementedError(self.add_object)
        return self.add_store.add_object(object)

    def add_objects(self, objects, progress=None):
        if self.add_store is None:
            raise NotImplementedError(self.add_object)
        return self.add_store.add_objects(objects, progress)

    @property
    def packs(self):
        ret = []
        for b in self.bases:
            ret.extend(b.packs)
        return ret

    def __iter__(self):
        done = set()
        for b in self.bases:
            for o_id in b:
                if o_id not in done:
                    yield o_id
                    done.add(o_id)

    def iterobjects_subset(self, shas: Iterable[bytes], *, allow_missing: bool=False) -> Iterator[ShaFile]:
        todo = set(shas)
        for b in self.bases:
            for o in b.iterobjects_subset(todo, allow_missing=True):
                yield o
                todo.remove(o.id)
        if todo and (not allow_missing):
            raise KeyError(o.id)

    def iter_unpacked_subset(self, shas: Iterable[bytes], *, include_comp=False, allow_missing: bool=False, convert_ofs_delta=True) -> Iterator[ShaFile]:
        todo = set(shas)
        for b in self.bases:
            for o in b.iter_unpacked_subset(todo, include_comp=include_comp, allow_missing=True, convert_ofs_delta=convert_ofs_delta):
                yield o
                todo.remove(o.id)
        if todo and (not allow_missing):
            raise KeyError(o.id)

    def get_raw(self, sha_id):
        for b in self.bases:
            try:
                return b.get_raw(sha_id)
            except KeyError:
                pass
        raise KeyError(sha_id)

    def contains_packed(self, sha):
        for b in self.bases:
            if b.contains_packed(sha):
                return True
        return False

    def contains_loose(self, sha):
        for b in self.bases:
            if b.contains_loose(sha):
                return True
        return False