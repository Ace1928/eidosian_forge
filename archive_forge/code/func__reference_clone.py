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