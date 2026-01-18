from __future__ import annotations
import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
class ImageFileDirectory_v2(MutableMapping):
    """This class represents a TIFF tag directory.  To speed things up, we
    don't decode tags unless they're asked for.

    Exposes a dictionary interface of the tags in the directory::

        ifd = ImageFileDirectory_v2()
        ifd[key] = 'Some Data'
        ifd.tagtype[key] = TiffTags.ASCII
        print(ifd[key])
        'Some Data'

    Individual values are returned as the strings or numbers, sequences are
    returned as tuples of the values.

    The tiff metadata type of each item is stored in a dictionary of
    tag types in
    :attr:`~PIL.TiffImagePlugin.ImageFileDirectory_v2.tagtype`. The types
    are read from a tiff file, guessed from the type added, or added
    manually.

    Data Structures:

        * ``self.tagtype = {}``

          * Key: numerical TIFF tag number
          * Value: integer corresponding to the data type from
            :py:data:`.TiffTags.TYPES`

          .. versionadded:: 3.0.0

    'Internal' data structures:

        * ``self._tags_v2 = {}``

          * Key: numerical TIFF tag number
          * Value: decoded data, as tuple for multiple values

        * ``self._tagdata = {}``

          * Key: numerical TIFF tag number
          * Value: undecoded byte string from file

        * ``self._tags_v1 = {}``

          * Key: numerical TIFF tag number
          * Value: decoded data in the v1 format

    Tags will be found in the private attributes ``self._tagdata``, and in
    ``self._tags_v2`` once decoded.

    ``self.legacy_api`` is a value for internal use, and shouldn't be changed
    from outside code. In cooperation with
    :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`, if ``legacy_api``
    is true, then decoded tags will be populated into both ``_tags_v1`` and
    ``_tags_v2``. ``_tags_v2`` will be used if this IFD is used in the TIFF
    save routine. Tags should be read from ``_tags_v1`` if
    ``legacy_api == true``.

    """

    def __init__(self, ifh=b'II*\x00\x00\x00\x00\x00', prefix=None, group=None):
        """Initialize an ImageFileDirectory.

        To construct an ImageFileDirectory from a real file, pass the 8-byte
        magic header to the constructor.  To only set the endianness, pass it
        as the 'prefix' keyword argument.

        :param ifh: One of the accepted magic headers (cf. PREFIXES); also sets
              endianness.
        :param prefix: Override the endianness of the file.
        """
        if not _accept(ifh):
            msg = f'not a TIFF file (header {repr(ifh)} not valid)'
            raise SyntaxError(msg)
        self._prefix = prefix if prefix is not None else ifh[:2]
        if self._prefix == MM:
            self._endian = '>'
        elif self._prefix == II:
            self._endian = '<'
        else:
            msg = 'not a TIFF IFD'
            raise SyntaxError(msg)
        self._bigtiff = ifh[2] == 43
        self.group = group
        self.tagtype = {}
        ' Dictionary of tag types '
        self.reset()
        self.next, = self._unpack('Q', ifh[8:]) if self._bigtiff else self._unpack('L', ifh[4:])
        self._legacy_api = False
    prefix = property(lambda self: self._prefix)
    offset = property(lambda self: self._offset)
    legacy_api = property(lambda self: self._legacy_api)

    @legacy_api.setter
    def legacy_api(self, value):
        msg = 'Not allowing setting of legacy api'
        raise Exception(msg)

    def reset(self):
        self._tags_v1 = {}
        self._tags_v2 = {}
        self._tagdata = {}
        self.tagtype = {}
        self._next = None
        self._offset = None

    def __str__(self):
        return str(dict(self))

    def named(self):
        """
        :returns: dict of name|key: value

        Returns the complete tag dictionary, with named tags where possible.
        """
        return {TiffTags.lookup(code, self.group).name: value for code, value in self.items()}

    def __len__(self):
        return len(set(self._tagdata) | set(self._tags_v2))

    def __getitem__(self, tag):
        if tag not in self._tags_v2:
            data = self._tagdata[tag]
            typ = self.tagtype[tag]
            size, handler = self._load_dispatch[typ]
            self[tag] = handler(self, data, self.legacy_api)
        val = self._tags_v2[tag]
        if self.legacy_api and (not isinstance(val, (tuple, bytes))):
            val = (val,)
        return val

    def __contains__(self, tag):
        return tag in self._tags_v2 or tag in self._tagdata

    def __setitem__(self, tag, value):
        self._setitem(tag, value, self.legacy_api)

    def _setitem(self, tag, value, legacy_api):
        basetypes = (Number, bytes, str)
        info = TiffTags.lookup(tag, self.group)
        values = [value] if isinstance(value, basetypes) else value
        if tag not in self.tagtype:
            if info.type:
                self.tagtype[tag] = info.type
            else:
                self.tagtype[tag] = TiffTags.UNDEFINED
                if all((isinstance(v, IFDRational) for v in values)):
                    self.tagtype[tag] = TiffTags.RATIONAL if all((v >= 0 for v in values)) else TiffTags.SIGNED_RATIONAL
                elif all((isinstance(v, int) for v in values)):
                    if all((0 <= v < 2 ** 16 for v in values)):
                        self.tagtype[tag] = TiffTags.SHORT
                    elif all((-2 ** 15 < v < 2 ** 15 for v in values)):
                        self.tagtype[tag] = TiffTags.SIGNED_SHORT
                    else:
                        self.tagtype[tag] = TiffTags.LONG if all((v >= 0 for v in values)) else TiffTags.SIGNED_LONG
                elif all((isinstance(v, float) for v in values)):
                    self.tagtype[tag] = TiffTags.DOUBLE
                elif all((isinstance(v, str) for v in values)):
                    self.tagtype[tag] = TiffTags.ASCII
                elif all((isinstance(v, bytes) for v in values)):
                    self.tagtype[tag] = TiffTags.BYTE
        if self.tagtype[tag] == TiffTags.UNDEFINED:
            values = [v.encode('ascii', 'replace') if isinstance(v, str) else v for v in values]
        elif self.tagtype[tag] == TiffTags.RATIONAL:
            values = [float(v) if isinstance(v, int) else v for v in values]
        is_ifd = self.tagtype[tag] == TiffTags.LONG and isinstance(values, dict)
        if not is_ifd:
            values = tuple((info.cvt_enum(value) for value in values))
        dest = self._tags_v1 if legacy_api else self._tags_v2
        if not is_ifd and (info.length == 1 or self.tagtype[tag] == TiffTags.BYTE or (info.length is None and len(values) == 1 and (not legacy_api))):
            if legacy_api and self.tagtype[tag] in [TiffTags.RATIONAL, TiffTags.SIGNED_RATIONAL]:
                values = (values,)
            try:
                dest[tag], = values
            except ValueError:
                warnings.warn(f'Metadata Warning, tag {tag} had too many entries: {len(values)}, expected 1')
                dest[tag] = values[0]
        else:
            dest[tag] = values

    def __delitem__(self, tag):
        self._tags_v2.pop(tag, None)
        self._tags_v1.pop(tag, None)
        self._tagdata.pop(tag, None)

    def __iter__(self):
        return iter(set(self._tagdata) | set(self._tags_v2))

    def _unpack(self, fmt, data):
        return struct.unpack(self._endian + fmt, data)

    def _pack(self, fmt, *values):
        return struct.pack(self._endian + fmt, *values)

    def _register_loader(idx, size):

        def decorator(func):
            from .TiffTags import TYPES
            if func.__name__.startswith('load_'):
                TYPES[idx] = func.__name__[5:].replace('_', ' ')
            _load_dispatch[idx] = (size, func)
            return func
        return decorator

    def _register_writer(idx):

        def decorator(func):
            _write_dispatch[idx] = func
            return func
        return decorator

    def _register_basic(idx_fmt_name):
        from .TiffTags import TYPES
        idx, fmt, name = idx_fmt_name
        TYPES[idx] = name
        size = struct.calcsize('=' + fmt)
        _load_dispatch[idx] = (size, lambda self, data, legacy_api=True: self._unpack(f'{len(data) // size}{fmt}', data))
        _write_dispatch[idx] = lambda self, *values: b''.join((self._pack(fmt, value) for value in values))
    list(map(_register_basic, [(TiffTags.SHORT, 'H', 'short'), (TiffTags.LONG, 'L', 'long'), (TiffTags.SIGNED_BYTE, 'b', 'signed byte'), (TiffTags.SIGNED_SHORT, 'h', 'signed short'), (TiffTags.SIGNED_LONG, 'l', 'signed long'), (TiffTags.FLOAT, 'f', 'float'), (TiffTags.DOUBLE, 'd', 'double'), (TiffTags.IFD, 'L', 'long'), (TiffTags.LONG8, 'Q', 'long8')]))

    @_register_loader(1, 1)
    def load_byte(self, data, legacy_api=True):
        return data

    @_register_writer(1)
    def write_byte(self, data):
        if isinstance(data, IFDRational):
            data = int(data)
        if isinstance(data, int):
            data = bytes((data,))
        return data

    @_register_loader(2, 1)
    def load_string(self, data, legacy_api=True):
        if data.endswith(b'\x00'):
            data = data[:-1]
        return data.decode('latin-1', 'replace')

    @_register_writer(2)
    def write_string(self, value):
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, bytes):
            value = value.encode('ascii', 'replace')
        return value + b'\x00'

    @_register_loader(5, 8)
    def load_rational(self, data, legacy_api=True):
        vals = self._unpack(f'{len(data) // 4}L', data)

        def combine(a, b):
            return (a, b) if legacy_api else IFDRational(a, b)
        return tuple((combine(num, denom) for num, denom in zip(vals[::2], vals[1::2])))

    @_register_writer(5)
    def write_rational(self, *values):
        return b''.join((self._pack('2L', *_limit_rational(frac, 2 ** 32 - 1)) for frac in values))

    @_register_loader(7, 1)
    def load_undefined(self, data, legacy_api=True):
        return data

    @_register_writer(7)
    def write_undefined(self, value):
        if isinstance(value, int):
            value = str(value).encode('ascii', 'replace')
        return value

    @_register_loader(10, 8)
    def load_signed_rational(self, data, legacy_api=True):
        vals = self._unpack(f'{len(data) // 4}l', data)

        def combine(a, b):
            return (a, b) if legacy_api else IFDRational(a, b)
        return tuple((combine(num, denom) for num, denom in zip(vals[::2], vals[1::2])))

    @_register_writer(10)
    def write_signed_rational(self, *values):
        return b''.join((self._pack('2l', *_limit_signed_rational(frac, 2 ** 31 - 1, -2 ** 31)) for frac in values))

    def _ensure_read(self, fp, size):
        ret = fp.read(size)
        if len(ret) != size:
            msg = f'Corrupt EXIF data.  Expecting to read {size} bytes but only got {len(ret)}. '
            raise OSError(msg)
        return ret

    def load(self, fp):
        self.reset()
        self._offset = fp.tell()
        try:
            tag_count = (self._unpack('Q', self._ensure_read(fp, 8)) if self._bigtiff else self._unpack('H', self._ensure_read(fp, 2)))[0]
            for i in range(tag_count):
                tag, typ, count, data = self._unpack('HHQ8s', self._ensure_read(fp, 20)) if self._bigtiff else self._unpack('HHL4s', self._ensure_read(fp, 12))
                tagname = TiffTags.lookup(tag, self.group).name
                typname = TYPES.get(typ, 'unknown')
                msg = f'tag: {tagname} ({tag}) - type: {typname} ({typ})'
                try:
                    unit_size, handler = self._load_dispatch[typ]
                except KeyError:
                    logger.debug('%s - unsupported type %s', msg, typ)
                    continue
                size = count * unit_size
                if size > (8 if self._bigtiff else 4):
                    here = fp.tell()
                    offset, = self._unpack('Q' if self._bigtiff else 'L', data)
                    msg += f' Tag Location: {here} - Data Location: {offset}'
                    fp.seek(offset)
                    data = ImageFile._safe_read(fp, size)
                    fp.seek(here)
                else:
                    data = data[:size]
                if len(data) != size:
                    warnings.warn(f'Possibly corrupt EXIF data.  Expecting to read {size} bytes but only got {len(data)}. Skipping tag {tag}')
                    logger.debug(msg)
                    continue
                if not data:
                    logger.debug(msg)
                    continue
                self._tagdata[tag] = data
                self.tagtype[tag] = typ
                msg += ' - value: ' + ('<table: %d bytes>' % size if size > 32 else repr(data))
                logger.debug(msg)
            self.next, = self._unpack('Q', self._ensure_read(fp, 8)) if self._bigtiff else self._unpack('L', self._ensure_read(fp, 4))
        except OSError as msg:
            warnings.warn(str(msg))
            return

    def tobytes(self, offset=0):
        result = self._pack('H', len(self._tags_v2))
        entries = []
        offset = offset + len(result) + len(self._tags_v2) * 12 + 4
        stripoffsets = None
        for tag, value in sorted(self._tags_v2.items()):
            if tag == STRIPOFFSETS:
                stripoffsets = len(entries)
            typ = self.tagtype.get(tag)
            logger.debug('Tag %s, Type: %s, Value: %s', tag, typ, repr(value))
            is_ifd = typ == TiffTags.LONG and isinstance(value, dict)
            if is_ifd:
                if self._endian == '<':
                    ifh = b'II*\x00\x08\x00\x00\x00'
                else:
                    ifh = b'MM\x00*\x00\x00\x00\x08'
                ifd = ImageFileDirectory_v2(ifh, group=tag)
                values = self._tags_v2[tag]
                for ifd_tag, ifd_value in values.items():
                    ifd[ifd_tag] = ifd_value
                data = ifd.tobytes(offset)
            else:
                values = value if isinstance(value, tuple) else (value,)
                data = self._write_dispatch[typ](self, *values)
            tagname = TiffTags.lookup(tag, self.group).name
            typname = 'ifd' if is_ifd else TYPES.get(typ, 'unknown')
            msg = f'save: {tagname} ({tag}) - type: {typname} ({typ})'
            msg += ' - value: ' + ('<table: %d bytes>' % len(data) if len(data) >= 16 else str(values))
            logger.debug(msg)
            if is_ifd:
                count = 1
            elif typ in [TiffTags.BYTE, TiffTags.ASCII, TiffTags.UNDEFINED]:
                count = len(data)
            else:
                count = len(values)
            if len(data) <= 4:
                entries.append((tag, typ, count, data.ljust(4, b'\x00'), b''))
            else:
                entries.append((tag, typ, count, self._pack('L', offset), data))
                offset += (len(data) + 1) // 2 * 2
        if stripoffsets is not None:
            tag, typ, count, value, data = entries[stripoffsets]
            if data:
                msg = 'multistrip support not yet implemented'
                raise NotImplementedError(msg)
            value = self._pack('L', self._unpack('L', value)[0] + offset)
            entries[stripoffsets] = (tag, typ, count, value, data)
        for tag, typ, count, value, data in entries:
            logger.debug('%s %s %s %s %s', tag, typ, count, repr(value), repr(data))
            result += self._pack('HHL4s', tag, typ, count, value)
        result += b'\x00\x00\x00\x00'
        for tag, typ, count, value, data in entries:
            result += data
            if len(data) & 1:
                result += b'\x00'
        return result

    def save(self, fp):
        if fp.tell() == 0:
            fp.write(self._prefix + self._pack('HL', 42, 8))
        offset = fp.tell()
        result = self.tobytes(offset)
        fp.write(result)
        return offset + len(result)