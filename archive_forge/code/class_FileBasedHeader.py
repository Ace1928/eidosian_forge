from __future__ import annotations
import io
import typing as ty
from copy import deepcopy
from urllib import request
from ._compression import COMPRESSION_ERRORS
from .fileholders import FileHolder, FileMap
from .filename_parser import TypesFilenamesError, _stringify_path, splitext_addext, types_filenames
from .openers import ImageOpener
class FileBasedHeader:
    """Template class to implement header protocol"""

    @classmethod
    def from_header(klass: type[HdrT], header: FileBasedHeader | ty.Mapping | None=None) -> HdrT:
        if header is None:
            return klass()
        if type(header) == klass:
            return header.copy()
        raise NotImplementedError(f'Header class requires a conversion from {klass} to {type(header)}')

    @classmethod
    def from_fileobj(klass: type[HdrT], fileobj: io.IOBase) -> HdrT:
        raise NotImplementedError

    def write_to(self, fileobj: io.IOBase) -> None:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __ne__(self, other: object) -> bool:
        return not self == other

    def copy(self: HdrT) -> HdrT:
        """Copy object to independent representation

        The copy should not be affected by any changes to the original
        object.
        """
        return deepcopy(self)