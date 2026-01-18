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
class ContentStream(DecodedStreamObject):
    """
    In order to be fast, this data structure can contain either:

    * raw data in ._data
    * parsed stream operations in ._operations.

    At any time, ContentStream object can either have both of those fields defined,
    or one field defined and the other set to None.

    These fields are "rebuilt" lazily, when accessed:

    * when .get_data() is called, if ._data is None, it is rebuilt from ._operations.
    * when .operations is called, if ._operations is None, it is rebuilt from ._data.

    Conversely, these fields can be invalidated:

    * when .set_data() is called, ._operations is set to None.
    * when .operations is set, ._data is set to None.
    """

    def __init__(self, stream: Any, pdf: Any, forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> None:
        self.pdf = pdf
        self._operations: List[Tuple[Any, Any]] = []
        if stream is None:
            super().set_data(b'')
        else:
            stream = stream.get_object()
            if isinstance(stream, ArrayObject):
                data = b''
                for s in stream:
                    data += b_(s.get_object().get_data())
                    if len(data) == 0 or data[-1] != b'\n':
                        data += b'\n'
                super().set_data(bytes(data))
            else:
                stream_data = stream.get_data()
                assert stream_data is not None
                super().set_data(b_(stream_data))
            self.forced_encoding = forced_encoding

    def clone(self, pdf_dest: Any, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'ContentStream':
        """
        Clone object into pdf_dest.

        Args:
            pdf_dest:
            force_duplicate:
            ignore_fields:

        Returns:
            The cloned ContentStream
        """
        try:
            if self.indirect_reference.pdf == pdf_dest and (not force_duplicate):
                return self
        except Exception:
            pass
        visited: Set[Tuple[int, int]] = set()
        d__ = cast('ContentStream', self._reference_clone(self.__class__(None, None), pdf_dest, force_duplicate))
        if ignore_fields is None:
            ignore_fields = []
        d__._clone(self, pdf_dest, force_duplicate, ignore_fields, visited)
        return d__

    def _clone(self, src: DictionaryObject, pdf_dest: PdfWriterProtocol, force_duplicate: bool, ignore_fields: Optional[Sequence[Union[str, int]]], visited: Set[Tuple[int, int]]) -> None:
        """
        Update the object from src.

        Args:
            src:
            pdf_dest:
            force_duplicate:
            ignore_fields:
        """
        src_cs = cast('ContentStream', src)
        super().set_data(b_(src_cs._data))
        self.pdf = pdf_dest
        self._operations = list(src_cs._operations)
        self.forced_encoding = src_cs.forced_encoding

    def _parse_content_stream(self, stream: StreamType) -> None:
        stream.seek(0, 0)
        operands: List[Union[int, str, PdfObject]] = []
        while True:
            peek = read_non_whitespace(stream)
            if peek == b'' or peek == 0:
                break
            stream.seek(-1, 1)
            if peek.isalpha() or peek in (b"'", b'"'):
                operator = read_until_regex(stream, NameObject.delimiter_pattern)
                if operator == b'BI':
                    assert operands == []
                    ii = self._read_inline_image(stream)
                    self._operations.append((ii, b'INLINE IMAGE'))
                else:
                    self._operations.append((operands, operator))
                    operands = []
            elif peek == b'%':
                while peek not in (b'\r', b'\n', b''):
                    peek = stream.read(1)
            else:
                operands.append(read_object(stream, None, self.forced_encoding))

    def _read_inline_image(self, stream: StreamType) -> Dict[str, Any]:
        settings = DictionaryObject()
        while True:
            tok = read_non_whitespace(stream)
            stream.seek(-1, 1)
            if tok == b'I':
                break
            key = read_object(stream, self.pdf)
            tok = read_non_whitespace(stream)
            stream.seek(-1, 1)
            value = read_object(stream, self.pdf)
            settings[key] = value
        tmp = stream.read(3)
        assert tmp[:2] == b'ID'
        data = BytesIO()
        while True:
            buf = stream.read(8192)
            if not buf:
                raise PdfReadError('Unexpected end of stream')
            loc = buf.find(b'E')
            if loc == -1:
                data.write(buf)
            else:
                data.write(buf[0:loc])
                stream.seek(loc - len(buf), 1)
                tok = stream.read(1)
                tok2 = stream.read(1)
                if tok2 != b'I':
                    stream.seek(-1, 1)
                    data.write(tok)
                    continue
                info = tok + tok2
                tok3 = stream.read(1)
                if tok3 not in WHITESPACES:
                    stream.seek(-2, 1)
                    data.write(tok)
                elif buf[loc - 1:loc] in WHITESPACES:
                    while tok3 in WHITESPACES:
                        tok3 = stream.read(1)
                    stream.seek(-1, 1)
                    break
                else:
                    while tok3 in WHITESPACES:
                        info += tok3
                        tok3 = stream.read(1)
                    stream.seek(-1, 1)
                    if tok3 == b'Q':
                        break
                    elif tok3 == b'E':
                        ope = stream.read(3)
                        stream.seek(-3, 1)
                        if ope == b'EMC':
                            break
                    else:
                        data.write(info)
        return {'settings': settings, 'data': data.getvalue()}

    def get_data(self) -> bytes:
        if not self._data:
            new_data = BytesIO()
            for operands, operator in self._operations:
                if operator == b'INLINE IMAGE':
                    new_data.write(b'BI')
                    dict_text = BytesIO()
                    operands['settings'].write_to_stream(dict_text)
                    new_data.write(dict_text.getvalue()[2:-2])
                    new_data.write(b'ID ')
                    new_data.write(operands['data'])
                    new_data.write(b'EI')
                else:
                    for op in operands:
                        op.write_to_stream(new_data)
                        new_data.write(b' ')
                    new_data.write(b_(operator))
                new_data.write(b'\n')
            self._data = new_data.getvalue()
        return b_(self._data)

    def set_data(self, data: bytes) -> None:
        super().set_data(data)
        self._operations = []

    @property
    def operations(self) -> List[Tuple[Any, Any]]:
        if not self._operations and self._data:
            self._parse_content_stream(BytesIO(b_(self._data)))
            self._data = b''
        return self._operations

    @operations.setter
    def operations(self, operations: List[Tuple[Any, Any]]) -> None:
        self._operations = operations
        self._data = b''

    def isolate_graphics_state(self) -> None:
        if self._operations:
            self._operations.insert(0, ([], 'q'))
            self._operations.append(([], 'Q'))
        elif self._data:
            self._data = b'q\n' + b_(self._data) + b'\nQ\n'

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if not self._data and self._operations:
            self.get_data()
        super().write_to_stream(stream, encryption_key)