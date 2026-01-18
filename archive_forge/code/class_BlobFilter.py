from binascii import b2a_hex
from pathlib import Path
from .util import pack, unpack
from git.objects import Blob
from typing import NamedTuple, Sequence, TYPE_CHECKING, Tuple, Union, cast, List
from git.types import PathLike
class BlobFilter:
    """
    Predicate to be used by iter_blobs allowing to filter only return blobs which
    match the given list of directories or files.

    The given paths are given relative to the repository.
    """
    __slots__ = ('paths',)

    def __init__(self, paths: Sequence[PathLike]) -> None:
        """
        :param paths:
            Tuple or list of paths which are either pointing to directories or
            to files relative to the current repository
        """
        self.paths = paths

    def __call__(self, stage_blob: Tuple[StageType, Blob]) -> bool:
        blob_pathlike: PathLike = stage_blob[1].path
        blob_path: Path = blob_pathlike if isinstance(blob_pathlike, Path) else Path(blob_pathlike)
        for pathlike in self.paths:
            path: Path = pathlike if isinstance(pathlike, Path) else Path(pathlike)
            filter_parts: List[str] = path.parts
            blob_parts: List[str] = blob_path.parts
            if len(filter_parts) > len(blob_parts):
                continue
            if all((i == j for i, j in zip(filter_parts, blob_parts))):
                return True
        return False