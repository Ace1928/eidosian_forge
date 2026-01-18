from git.exc import WorkTreeRepositoryUnsupported
from git.util import LazyMixin, join_path_native, stream_copy, bin_to_hex
import gitdb.typ as dbtyp
import os.path as osp
from .util import get_object_type_by_name
from typing import Any, TYPE_CHECKING, Union
from git.types import PathLike, Commit_ish, Lit_commit_ish
class Object(LazyMixin):
    """An Object which may be Blobs, Trees, Commits and Tags."""
    NULL_HEX_SHA = '0' * 40
    NULL_BIN_SHA = b'\x00' * 20
    TYPES = (dbtyp.str_blob_type, dbtyp.str_tree_type, dbtyp.str_commit_type, dbtyp.str_tag_type)
    __slots__ = ('repo', 'binsha', 'size')
    type: Union[Lit_commit_ish, None] = None

    def __init__(self, repo: 'Repo', binsha: bytes):
        """Initialize an object by identifying it by its binary sha.
        All keyword arguments will be set on demand if None.

        :param repo: repository this object is located in

        :param binsha: 20 byte SHA1
        """
        super().__init__()
        self.repo = repo
        self.binsha = binsha
        assert len(binsha) == 20, 'Require 20 byte binary sha, got %r, len = %i' % (binsha, len(binsha))

    @classmethod
    def new(cls, repo: 'Repo', id: Union[str, 'Reference']) -> Commit_ish:
        """
        :return: New :class:`Object`` instance of a type appropriate to the object type
            behind `id`. The id of the newly created object will be a binsha even though
            the input id may have been a Reference or Rev-Spec.

        :param id: reference, rev-spec, or hexsha

        :note: This cannot be a ``__new__`` method as it would always call
            :meth:`__init__` with the input id which is not necessarily a binsha.
        """
        return repo.rev_parse(str(id))

    @classmethod
    def new_from_sha(cls, repo: 'Repo', sha1: bytes) -> Commit_ish:
        """
        :return: new object instance of a type appropriate to represent the given
            binary sha1

        :param sha1: 20 byte binary sha1
        """
        if sha1 == cls.NULL_BIN_SHA:
            return get_object_type_by_name(b'commit')(repo, sha1)
        oinfo = repo.odb.info(sha1)
        inst = get_object_type_by_name(oinfo.type)(repo, oinfo.binsha)
        inst.size = oinfo.size
        return inst

    def _set_cache_(self, attr: str) -> None:
        """Retrieve object information."""
        if attr == 'size':
            oinfo = self.repo.odb.info(self.binsha)
            self.size = oinfo.size
        else:
            super()._set_cache_(attr)

    def __eq__(self, other: Any) -> bool:
        """:return: True if the objects have the same SHA1"""
        if not hasattr(other, 'binsha'):
            return False
        return self.binsha == other.binsha

    def __ne__(self, other: Any) -> bool:
        """:return: True if the objects do not have the same SHA1"""
        if not hasattr(other, 'binsha'):
            return True
        return self.binsha != other.binsha

    def __hash__(self) -> int:
        """:return: Hash of our id allowing objects to be used in dicts and sets"""
        return hash(self.binsha)

    def __str__(self) -> str:
        """:return: string of our SHA1 as understood by all git commands"""
        return self.hexsha

    def __repr__(self) -> str:
        """:return: string with pythonic representation of our object"""
        return '<git.%s "%s">' % (self.__class__.__name__, self.hexsha)

    @property
    def hexsha(self) -> str:
        """:return: 40 byte hex version of our 20 byte binary sha"""
        return bin_to_hex(self.binsha).decode('ascii')

    @property
    def data_stream(self) -> 'OStream':
        """
        :return: File Object compatible stream to the uncompressed raw data of the object

        :note: Returned streams must be read in order.
        """
        return self.repo.odb.stream(self.binsha)

    def stream_data(self, ostream: 'OStream') -> 'Object':
        """Write our data directly to the given output stream.

        :param ostream: File object compatible stream object.
        :return: self
        """
        istream = self.repo.odb.stream(self.binsha)
        stream_copy(istream, ostream)
        return self