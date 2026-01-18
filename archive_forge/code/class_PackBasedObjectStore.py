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
class PackBasedObjectStore(BaseObjectStore):

    def __init__(self, pack_compression_level=-1) -> None:
        self._pack_cache: Dict[str, Pack] = {}
        self.pack_compression_level = pack_compression_level

    def add_pack(self) -> Tuple[BytesIO, Callable[[], None], Callable[[], None]]:
        """Add a new pack to this object store."""
        raise NotImplementedError(self.add_pack)

    def add_pack_data(self, count: int, unpacked_objects: Iterator[UnpackedObject], progress=None) -> None:
        """Add pack data to this object store.

        Args:
          count: Number of items to add
          pack_data: Iterator over pack data tuples
        """
        if count == 0:
            return
        f, commit, abort = self.add_pack()
        try:
            write_pack_data(f.write, unpacked_objects, num_records=count, progress=progress, compression_level=self.pack_compression_level)
        except BaseException:
            abort()
            raise
        else:
            return commit()

    @property
    def alternates(self):
        return []

    def contains_packed(self, sha):
        """Check if a particular object is present by SHA1 and is packed.

        This does not check alternates.
        """
        for pack in self.packs:
            try:
                if sha in pack:
                    return True
            except PackFileDisappeared:
                pass
        return False

    def __contains__(self, sha) -> bool:
        """Check if a particular object is present by SHA1.

        This method makes no distinction between loose and packed objects.
        """
        if self.contains_packed(sha) or self.contains_loose(sha):
            return True
        for alternate in self.alternates:
            if sha in alternate:
                return True
        return False

    def _add_cached_pack(self, base_name, pack):
        """Add a newly appeared pack to the cache by path."""
        prev_pack = self._pack_cache.get(base_name)
        if prev_pack is not pack:
            self._pack_cache[base_name] = pack
            if prev_pack:
                prev_pack.close()

    def generate_pack_data(self, have, want, shallow=None, progress=None, ofs_delta=True) -> Tuple[int, Iterator[UnpackedObject]]:
        """Generate pack data objects for a set of wants/haves.

        Args:
          have: List of SHA1s of objects that should not be sent
          want: List of SHA1s of objects that should be sent
          shallow: Set of shallow commit SHA1s to skip
          ofs_delta: Whether OFS deltas can be included
          progress: Optional progress reporting method
        """
        missing_objects = MissingObjectFinder(self, haves=have, wants=want, shallow=shallow, progress=progress)
        remote_has = missing_objects.get_remote_has()
        object_ids = list(missing_objects)
        return (len(object_ids), generate_unpacked_objects(cast(PackedObjectContainer, self), object_ids, progress=progress, ofs_delta=ofs_delta, other_haves=remote_has))

    def _clear_cached_packs(self):
        pack_cache = self._pack_cache
        self._pack_cache = {}
        while pack_cache:
            name, pack = pack_cache.popitem()
            pack.close()

    def _iter_cached_packs(self):
        return self._pack_cache.values()

    def _update_pack_cache(self):
        raise NotImplementedError(self._update_pack_cache)

    def close(self):
        self._clear_cached_packs()

    @property
    def packs(self):
        """List with pack objects."""
        return list(self._iter_cached_packs()) + list(self._update_pack_cache())

    def _iter_alternate_objects(self):
        """Iterate over the SHAs of all the objects in alternate stores."""
        for alternate in self.alternates:
            yield from alternate

    def _iter_loose_objects(self):
        """Iterate over the SHAs of all loose objects."""
        raise NotImplementedError(self._iter_loose_objects)

    def _get_loose_object(self, sha):
        raise NotImplementedError(self._get_loose_object)

    def _remove_loose_object(self, sha):
        raise NotImplementedError(self._remove_loose_object)

    def _remove_pack(self, name):
        raise NotImplementedError(self._remove_pack)

    def pack_loose_objects(self):
        """Pack loose objects.

        Returns: Number of objects packed
        """
        objects = set()
        for sha in self._iter_loose_objects():
            objects.add((self._get_loose_object(sha), None))
        self.add_objects(list(objects))
        for obj, path in objects:
            self._remove_loose_object(obj.id)
        return len(objects)

    def repack(self):
        """Repack the packs in this repository.

        Note that this implementation is fairly naive and currently keeps all
        objects in memory while it repacks.
        """
        loose_objects = set()
        for sha in self._iter_loose_objects():
            loose_objects.add(self._get_loose_object(sha))
        objects = {(obj, None) for obj in loose_objects}
        old_packs = {p.name(): p for p in self.packs}
        for name, pack in old_packs.items():
            objects.update(((obj, None) for obj in pack.iterobjects()))
        consolidated = self.add_objects(objects)
        old_packs.pop(consolidated.name(), None)
        for obj in loose_objects:
            self._remove_loose_object(obj.id)
        for name, pack in old_packs.items():
            self._remove_pack(pack)
        self._update_pack_cache()
        return len(objects)

    def __iter__(self):
        """Iterate over the SHAs that are present in this store."""
        self._update_pack_cache()
        for pack in self._iter_cached_packs():
            try:
                yield from pack
            except PackFileDisappeared:
                pass
        yield from self._iter_loose_objects()
        yield from self._iter_alternate_objects()

    def contains_loose(self, sha):
        """Check if a particular object is present by SHA1 and is loose.

        This does not check alternates.
        """
        return self._get_loose_object(sha) is not None

    def get_raw(self, name):
        """Obtain the raw fulltext for an object.

        Args:
          name: sha for the object.
        Returns: tuple with numeric type and object contents.
        """
        if name == ZERO_SHA:
            raise KeyError(name)
        if len(name) == 40:
            sha = hex_to_sha(name)
            hexsha = name
        elif len(name) == 20:
            sha = name
            hexsha = None
        else:
            raise AssertionError(f'Invalid object name {name!r}')
        for pack in self._iter_cached_packs():
            try:
                return pack.get_raw(sha)
            except (KeyError, PackFileDisappeared):
                pass
        if hexsha is None:
            hexsha = sha_to_hex(name)
        ret = self._get_loose_object(hexsha)
        if ret is not None:
            return (ret.type_num, ret.as_raw_string())
        for pack in self._update_pack_cache():
            try:
                return pack.get_raw(sha)
            except KeyError:
                pass
        for alternate in self.alternates:
            try:
                return alternate.get_raw(hexsha)
            except KeyError:
                pass
        raise KeyError(hexsha)

    def iter_unpacked_subset(self, shas, *, include_comp=False, allow_missing: bool=False, convert_ofs_delta: bool=True) -> Iterator[ShaFile]:
        todo: Set[bytes] = set(shas)
        for p in self._iter_cached_packs():
            for unpacked in p.iter_unpacked_subset(todo, include_comp=include_comp, allow_missing=True, convert_ofs_delta=convert_ofs_delta):
                yield unpacked
                hexsha = sha_to_hex(unpacked.sha())
                todo.remove(hexsha)
        for p in self._update_pack_cache():
            for unpacked in p.iter_unpacked_subset(todo, include_comp=include_comp, allow_missing=True, convert_ofs_delta=convert_ofs_delta):
                yield unpacked
                hexsha = sha_to_hex(unpacked.sha())
                todo.remove(hexsha)
        for alternate in self.alternates:
            for unpacked in alternate.iter_unpacked_subset(todo, include_comp=include_comp, allow_missing=True, convert_ofs_delta=convert_ofs_delta):
                yield unpacked
                hexsha = sha_to_hex(unpacked.sha())
                todo.remove(hexsha)

    def iterobjects_subset(self, shas: Iterable[bytes], *, allow_missing: bool=False) -> Iterator[ShaFile]:
        todo: Set[bytes] = set(shas)
        for p in self._iter_cached_packs():
            for o in p.iterobjects_subset(todo, allow_missing=True):
                yield o
                todo.remove(o.id)
        for p in self._update_pack_cache():
            for o in p.iterobjects_subset(todo, allow_missing=True):
                yield o
                todo.remove(o.id)
        for alternate in self.alternates:
            for o in alternate.iterobjects_subset(todo, allow_missing=True):
                yield o
                todo.remove(o.id)
        for oid in todo:
            o = self._get_loose_object(oid)
            if o is not None:
                yield o
            elif not allow_missing:
                raise KeyError(oid)

    def get_unpacked_object(self, sha1: bytes, *, include_comp: bool=False) -> UnpackedObject:
        """Obtain the unpacked object.

        Args:
          sha1: sha for the object.
        """
        if sha1 == ZERO_SHA:
            raise KeyError(sha1)
        if len(sha1) == 40:
            sha = hex_to_sha(sha1)
            hexsha = sha1
        elif len(sha1) == 20:
            sha = sha1
            hexsha = None
        else:
            raise AssertionError(f'Invalid object sha1 {sha1!r}')
        for pack in self._iter_cached_packs():
            try:
                return pack.get_unpacked_object(sha, include_comp=include_comp)
            except (KeyError, PackFileDisappeared):
                pass
        if hexsha is None:
            hexsha = sha_to_hex(sha1)
        for pack in self._update_pack_cache():
            try:
                return pack.get_unpacked_object(sha, include_comp=include_comp)
            except KeyError:
                pass
        for alternate in self.alternates:
            try:
                return alternate.get_unpacked_object(hexsha, include_comp=include_comp)
            except KeyError:
                pass
        raise KeyError(hexsha)

    def add_objects(self, objects: Sequence[Tuple[ShaFile, Optional[str]]], progress: Optional[Callable[[str], None]]=None) -> None:
        """Add a set of objects to this object store.

        Args:
          objects: Iterable over (object, path) tuples, should support
            __len__.
        Returns: Pack object of the objects written.
        """
        count = len(objects)
        record_iter = (full_unpacked_object(o) for o, p in objects)
        return self.add_pack_data(count, record_iter, progress=progress)