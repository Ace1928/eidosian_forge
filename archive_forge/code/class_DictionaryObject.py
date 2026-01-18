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
class DictionaryObject(Dict[Any, Any], PdfObject):

    def clone(self, pdf_dest: PdfWriterProtocol, force_duplicate: bool=False, ignore_fields: Optional[Sequence[Union[str, int]]]=()) -> 'DictionaryObject':
        """Clone object into pdf_dest."""
        try:
            if self.indirect_reference.pdf == pdf_dest and (not force_duplicate):
                return self
        except Exception:
            pass
        visited: Set[Tuple[int, int]] = set()
        d__ = cast('DictionaryObject', self._reference_clone(self.__class__(), pdf_dest, force_duplicate))
        if ignore_fields is None:
            ignore_fields = []
        if len(d__.keys()) == 0:
            d__._clone(self, pdf_dest, force_duplicate, ignore_fields, visited)
        return d__

    def _clone(self, src: 'DictionaryObject', pdf_dest: PdfWriterProtocol, force_duplicate: bool, ignore_fields: Optional[Sequence[Union[str, int]]], visited: Set[Tuple[int, int]]) -> None:
        """
        Update the object from src.

        Args:
            src: "DictionaryObject":
            pdf_dest:
            force_duplicate:
            ignore_fields:
        """
        x = 0
        assert ignore_fields is not None
        ignore_fields = list(ignore_fields)
        while x < len(ignore_fields):
            if isinstance(ignore_fields[x], int):
                if cast(int, ignore_fields[x]) <= 0:
                    del ignore_fields[x]
                    del ignore_fields[x]
                    continue
                else:
                    ignore_fields[x] -= 1
            x += 1
        if any((field not in ignore_fields and field in src and isinstance(src.raw_get(field), IndirectObject) and isinstance(src[field], DictionaryObject) and (src.get('/Type', None) is None or cast(DictionaryObject, src[field]).get('/Type', None) is None or src.get('/Type', None) == cast(DictionaryObject, src[field]).get('/Type', None)) for field in ['/Next', '/Prev', '/N', '/V'])):
            ignore_fields = list(ignore_fields)
            for lst in (('/Next', '/Prev'), ('/N', '/V')):
                for k in lst:
                    objs = []
                    if k in src and k not in self and isinstance(src.raw_get(k), IndirectObject) and isinstance(src[k], DictionaryObject) and (src.get('/Type', None) is None or cast(DictionaryObject, src[k]).get('/Type', None) is None or src.get('/Type', None) == cast(DictionaryObject, src[k]).get('/Type', None)):
                        cur_obj: Optional[DictionaryObject] = cast('DictionaryObject', src[k])
                        prev_obj: Optional[DictionaryObject] = self
                        while cur_obj is not None:
                            clon = cast('DictionaryObject', cur_obj._reference_clone(cur_obj.__class__(), pdf_dest, force_duplicate))
                            if clon.indirect_reference is not None:
                                idnum = clon.indirect_reference.idnum
                                generation = clon.indirect_reference.generation
                                if (idnum, generation) in visited:
                                    cur_obj = None
                                    break
                                visited.add((idnum, generation))
                            objs.append((cur_obj, clon))
                            assert prev_obj is not None
                            prev_obj[NameObject(k)] = clon.indirect_reference
                            prev_obj = clon
                            try:
                                if cur_obj == src:
                                    cur_obj = None
                                else:
                                    cur_obj = cast('DictionaryObject', cur_obj[k])
                            except Exception:
                                cur_obj = None
                        for s, c in objs:
                            c._clone(s, pdf_dest, force_duplicate, ignore_fields, visited)
        for k, v in src.items():
            if k not in ignore_fields:
                if isinstance(v, StreamObject):
                    if not hasattr(v, 'indirect_reference'):
                        v.indirect_reference = None
                    vv = v.clone(pdf_dest, force_duplicate, ignore_fields)
                    assert vv.indirect_reference is not None
                    self[k.clone(pdf_dest)] = vv.indirect_reference
                elif k not in self:
                    self[NameObject(k)] = v.clone(pdf_dest, force_duplicate, ignore_fields) if hasattr(v, 'clone') else v

    def raw_get(self, key: Any) -> Any:
        return dict.__getitem__(self, key)

    def get_inherited(self, key: str, default: Any=None) -> Any:
        """
        Returns the value of a key or from the parent if not found.
        If not found returns default.

        Args:
            key: string identifying the field to return

            default: default value to return

        Returns:
            Current key or inherited one, otherwise default value.
        """
        if key in self:
            return self[key]
        try:
            if '/Parent' not in self:
                return default
            raise KeyError('not present')
        except KeyError:
            return cast('DictionaryObject', self['/Parent'].get_object()).get_inherited(key, default)

    def __setitem__(self, key: Any, value: Any) -> Any:
        if not isinstance(key, PdfObject):
            raise ValueError('key must be PdfObject')
        if not isinstance(value, PdfObject):
            raise ValueError('value must be PdfObject')
        return dict.__setitem__(self, key, value)

    def setdefault(self, key: Any, value: Optional[Any]=None) -> Any:
        if not isinstance(key, PdfObject):
            raise ValueError('key must be PdfObject')
        if not isinstance(value, PdfObject):
            raise ValueError('value must be PdfObject')
        return dict.setdefault(self, key, value)

    def __getitem__(self, key: Any) -> PdfObject:
        return dict.__getitem__(self, key).get_object()

    @property
    def xmp_metadata(self) -> Optional[XmpInformationProtocol]:
        """
        Retrieve XMP (Extensible Metadata Platform) data relevant to the this
        object, if available.

        See Table 347 — Additional entries in a metadata stream dictionary.

        Returns:
          Returns a :class:`~pypdf.xmp.XmpInformation` instance
          that can be used to access XMP metadata from the document.  Can also
          return None if no metadata was found on the document root.
        """
        from ..xmp import XmpInformation
        metadata = self.get('/Metadata', None)
        if metadata is None:
            return None
        metadata = metadata.get_object()
        if not isinstance(metadata, XmpInformation):
            metadata = XmpInformation(metadata)
            self[NameObject('/Metadata')] = metadata
        return metadata

    def write_to_stream(self, stream: StreamType, encryption_key: Union[None, str, bytes]=None) -> None:
        if encryption_key is not None:
            deprecate_no_replacement('the encryption_key parameter of write_to_stream', '5.0.0')
        stream.write(b'<<\n')
        for key, value in list(self.items()):
            if len(key) > 2 and key[1] == '%' and (key[-1] == '%'):
                continue
            key.write_to_stream(stream, encryption_key)
            stream.write(b' ')
            value.write_to_stream(stream)
            stream.write(b'\n')
        stream.write(b'>>')

    @staticmethod
    def read_from_stream(stream: StreamType, pdf: Optional[PdfReaderProtocol], forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> 'DictionaryObject':

        def get_next_obj_pos(p: int, p1: int, rem_gens: List[int], pdf: PdfReaderProtocol) -> int:
            out = p1
            for gen in rem_gens:
                loc = pdf.xref[gen]
                try:
                    out = min(out, min([x for x in loc.values() if p < x <= p1]))
                except ValueError:
                    pass
            return out

        def read_unsized_from_stream(stream: StreamType, pdf: PdfReaderProtocol) -> bytes:
            eon = get_next_obj_pos(stream.tell(), 2 ** 32, list(pdf.xref), pdf) - 1
            curr = stream.tell()
            rw = stream.read(eon - stream.tell())
            p = rw.find(b'endstream')
            if p < 0:
                raise PdfReadError(f"Unable to find 'endstream' marker for obj starting at {curr}.")
            stream.seek(curr + p + 9)
            return rw[:p - 1]
        tmp = stream.read(2)
        if tmp != b'<<':
            raise PdfReadError(f"Dictionary read error at byte {hex(stream.tell())}: stream must begin with '<<'")
        data: Dict[Any, Any] = {}
        while True:
            tok = read_non_whitespace(stream)
            if tok == b'\x00':
                continue
            elif tok == b'%':
                stream.seek(-1, 1)
                skip_over_comment(stream)
                continue
            if not tok:
                raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
            if tok == b'>':
                stream.read(1)
                break
            stream.seek(-1, 1)
            try:
                key = read_object(stream, pdf)
                tok = read_non_whitespace(stream)
                stream.seek(-1, 1)
                value = read_object(stream, pdf, forced_encoding)
            except Exception as exc:
                if pdf is not None and pdf.strict:
                    raise PdfReadError(exc.__repr__())
                logger_warning(exc.__repr__(), __name__)
                retval = DictionaryObject()
                retval.update(data)
                return retval
            if not data.get(key):
                data[key] = value
            else:
                msg = f'Multiple definitions in dictionary at byte {hex(stream.tell())} for key {key}'
                if pdf is not None and pdf.strict:
                    raise PdfReadError(msg)
                logger_warning(msg, __name__)
        pos = stream.tell()
        s = read_non_whitespace(stream)
        if s == b's' and stream.read(5) == b'tream':
            eol = stream.read(1)
            while eol == b' ':
                eol = stream.read(1)
            if eol not in (b'\n', b'\r'):
                raise PdfStreamError('Stream data must be followed by a newline')
            if eol == b'\r' and stream.read(1) != b'\n':
                stream.seek(-1, 1)
            if SA.LENGTH not in data:
                if pdf is not None and pdf.strict:
                    raise PdfStreamError('Stream length not defined')
                else:
                    logger_warning(f'Stream length not defined @pos={stream.tell()}', __name__)
                data[NameObject(SA.LENGTH)] = NumberObject(-1)
            length = data[SA.LENGTH]
            if isinstance(length, IndirectObject):
                t = stream.tell()
                assert pdf is not None
                length = pdf.get_object(length)
                stream.seek(t, 0)
            if length is None:
                length = -1
            pstart = stream.tell()
            if length > 0:
                data['__streamdata__'] = stream.read(length)
            else:
                data['__streamdata__'] = read_until_regex(stream, re.compile(b'endstream'))
            e = read_non_whitespace(stream)
            ndstream = stream.read(8)
            if e + ndstream != b'endstream':
                pos = stream.tell()
                stream.seek(-10, 1)
                end = stream.read(9)
                if end == b'endstream':
                    data['__streamdata__'] = data['__streamdata__'][:-1]
                elif pdf is not None and (not pdf.strict):
                    stream.seek(pstart, 0)
                    data['__streamdata__'] = read_unsized_from_stream(stream, pdf)
                    pos = stream.tell()
                else:
                    stream.seek(pos, 0)
                    raise PdfReadError(f"Unable to find 'endstream' marker after stream at byte {hex(stream.tell())} (nd='{ndstream!r}', end='{end!r}').")
        else:
            stream.seek(pos, 0)
        if '__streamdata__' in data:
            return StreamObject.initialize_from_dictionary(data)
        else:
            retval = DictionaryObject()
            retval.update(data)
            return retval