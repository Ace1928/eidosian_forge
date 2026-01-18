import logging
import re
import sys
from io import BytesIO
from typing import (
from .._protocols import PdfReaderProtocol, PdfWriterProtocol, XmpInformationProtocol
from .._utils import (
from ..constants import (
from ..constants import FilterTypes as FT
from ..constants import StreamAttributes as SA
from ..constants import TypArguments as TA
from ..constants import TypFitArguments as TF
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
from ._base import (
from ._fit import Fit
from ._utils import read_hex_string_from_stream, read_string_from_stream
class ArrayObject(List[Any], PdfObject):

    def clone(self, pdf_dest: PdfWriterProtocol, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'ArrayObject':
        """Clone object into pdf_dest."""
        try:
            if self.indirect_reference.pdf == pdf_dest and (not force_duplicate):
                return self
        except Exception:
            pass
        arr = cast('ArrayObject', self._reference_clone(ArrayObject(), pdf_dest, force_duplicate))
        for data in self:
            if isinstance(data, StreamObject):
                dup = data._reference_clone(data.clone(pdf_dest, force_duplicate, ignore_fields), pdf_dest, force_duplicate)
                arr.append(dup.indirect_reference)
            elif hasattr(data, 'clone'):
                arr.append(data.clone(pdf_dest, force_duplicate, ignore_fields))
            else:
                arr.append(data)
        return arr

    def items(self) -> Iterable[Any]:
        """Emulate DictionaryObject.items for a list (index, object)."""
        return enumerate(self)

    def _to_lst(self, lst: Any) -> List[Any]:
        if isinstance(lst, (list, tuple, set)):
            pass
        elif isinstance(lst, PdfObject):
            lst = [lst]
        elif isinstance(lst, str):
            if lst[0] == '/':
                lst = [NameObject(lst)]
            else:
                lst = [TextStringObject(lst)]
        elif isinstance(lst, bytes):
            lst = [ByteStringObject(lst)]
        else:
            lst = [lst]
        return lst

    def __add__(self, lst: Any) -> 'ArrayObject':
        """
        Allow extension by adding list or add one element only

        Args:
            lst: any list, tuples are extended the list.
            other types(numbers,...) will be appended.
            if str is passed it will be converted into TextStringObject
            or NameObject (if starting with "/")
            if bytes is passed it will be converted into ByteStringObject

        Returns:
            ArrayObject with all elements
        """
        temp = ArrayObject(self)
        temp.extend(self._to_lst(lst))
        return temp

    def __iadd__(self, lst: Any) -> Self:
        """
         Allow extension by adding list or add one element only

        Args:
            lst: any list, tuples are extended the list.
            other types(numbers,...) will be appended.
            if str is passed it will be converted into TextStringObject
            or NameObject (if starting with "/")
            if bytes is passed it will be converted into ByteStringObject
        """
        self.extend(self._to_lst(lst))
        return self

    def __isub__(self, lst: Any) -> Self:
        """Allow to remove items"""
        for x in self._to_lst(lst):
            try:
                x = self.index(x)
                del self[x]
            except ValueError:
                pass
        return self

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        stream.write(b'[')
        for data in self:
            stream.write(b' ')
            data.write_to_stream(stream)
        stream.write(b' ]')

    @staticmethod
    def read_from_stream(stream: StreamType, pdf: Optional[PdfReaderProtocol], forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> 'ArrayObject':
        arr = ArrayObject()
        tmp = stream.read(1)
        if tmp != b'[':
            raise PdfReadError('Could not read array')
        while True:
            tok = stream.read(1)
            while tok.isspace():
                tok = stream.read(1)
            stream.seek(-1, 1)
            peek_ahead = stream.read(1)
            if peek_ahead == b']':
                break
            stream.seek(-1, 1)
            arr.append(read_object(stream, pdf, forced_encoding))
        return arr