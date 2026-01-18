from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
class IndexEntry(BaseIndexEntry):
    """Allows convenient access to IndexEntry data without completely unpacking it.

    Attributes usually accessed often are cached in the tuple whereas others are
    unpacked on demand.

    See the properties for a mapping between names and tuple indices.
    """

    @property
    def ctime(self) -> Tuple[int, int]:
        """
        :return:
            Tuple(int_time_seconds_since_epoch, int_nano_seconds) of the
            file's creation time
        """
        return cast(Tuple[int, int], unpack('>LL', self.ctime_bytes))

    @property
    def mtime(self) -> Tuple[int, int]:
        """See ctime property, but returns modification time."""
        return cast(Tuple[int, int], unpack('>LL', self.mtime_bytes))

    @classmethod
    def from_base(cls, base: 'BaseIndexEntry') -> 'IndexEntry':
        """
        :return:
            Minimal entry as created from the given BaseIndexEntry instance.
            Missing values will be set to null-like values.

        :param base: Instance of type :class:`BaseIndexEntry`
        """
        time = pack('>LL', 0, 0)
        return IndexEntry((base.mode, base.binsha, base.flags, base.path, time, time, 0, 0, 0, 0, 0))

    @classmethod
    def from_blob(cls, blob: Blob, stage: int=0) -> 'IndexEntry':
        """:return: Minimal entry resembling the given blob object"""
        time = pack('>LL', 0, 0)
        return IndexEntry((blob.mode, blob.binsha, stage << CE_STAGESHIFT, blob.path, time, time, 0, 0, 0, 0, blob.size))