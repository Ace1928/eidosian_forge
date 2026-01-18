import binascii
import codecs
import hashlib
import re
from binascii import unhexlify
from math import log10
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence, Union, cast
from .._codecs import _pdfdoc_encoding_rev
from .._protocols import PdfObjectProtocol, PdfWriterProtocol
from .._utils import (
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
class PdfObject(PdfObjectProtocol):
    hash_func: Callable[..., 'hashlib._Hash'] = hashlib.sha1
    indirect_reference: Optional['IndirectObject']

    def hash_value_data(self) -> bytes:
        return ('%s' % self).encode()

    def hash_value(self) -> bytes:
        return ('%s:%s' % (self.__class__.__name__, self.hash_func(self.hash_value_data()).hexdigest())).encode()

    def clone(self, pdf_dest: PdfWriterProtocol, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'PdfObject':
        """
        Clone object into pdf_dest (PdfWriterProtocol which is an interface for PdfWriter).

        By default, this method will call ``_reference_clone`` (see ``_reference``).


        Args:
          pdf_dest: Target to clone to.
          force_duplicate: By default, if the object has already been cloned and referenced,
            the copy will be returned; when ``True``, a new copy will be created.
            (Default value = ``False``)
          ignore_fields: List/tuple of field names (for dictionaries) that will be ignored
            during cloning (applies to children duplication as well). If fields are to be
            considered for a limited number of levels, you have to add it as integer, for
            example ``[1,"/B","/TOTO"]`` means that ``"/B"`` will be ignored at the first
            level only but ``"/TOTO"`` on all levels.

        Returns:
          The cloned PdfObject
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement .clone so far')

    def _reference_clone(self, clone: Any, pdf_dest: PdfWriterProtocol, force_duplicate: bool=False) -> PdfObjectProtocol:
        """
        Reference the object within the _objects of pdf_dest only if
        indirect_reference attribute exists (which means the objects was
        already identified in xref/xobjstm) if object has been already
        referenced do nothing.

        Args:
          clone:
          pdf_dest:

        Returns:
          The clone
        """
        try:
            if not force_duplicate and clone.indirect_reference.pdf == pdf_dest:
                return clone
        except Exception:
            pass
        try:
            ind = self.indirect_reference
        except AttributeError:
            return clone
        i = len(pdf_dest._objects) + 1
        if ind is not None:
            if id(ind.pdf) not in pdf_dest._id_translated:
                pdf_dest._id_translated[id(ind.pdf)] = {}
                pdf_dest._id_translated[id(ind.pdf)]['PreventGC'] = ind.pdf
            if not force_duplicate and ind.idnum in pdf_dest._id_translated[id(ind.pdf)]:
                obj = pdf_dest.get_object(pdf_dest._id_translated[id(ind.pdf)][ind.idnum])
                assert obj is not None
                return obj
            pdf_dest._id_translated[id(ind.pdf)][ind.idnum] = i
        pdf_dest._objects.append(clone)
        clone.indirect_reference = IndirectObject(i, 0, pdf_dest)
        return clone

    def get_object(self) -> Optional['PdfObject']:
        """Resolve indirect references."""
        return self

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        raise NotImplementedError