from __future__ import absolute_import, division, print_function
import bz2
import hashlib
import logging
import os
import re
import struct
import sys
import types
import zlib
from io import BytesIO
class BsdfSerializer(object):
    """Instances of this class represent a BSDF encoder/decoder.

    It acts as a placeholder for a set of extensions and encoding/decoding
    options. Use this to predefine extensions and options for high
    performance encoding/decoding. For general use, see the functions
    `save()`, `encode()`, `load()`, and `decode()`.

    This implementation of BSDF supports streaming lists (keep adding
    to a list after writing the main file), lazy loading of blobs, and
    in-place editing of blobs (for streams opened with a+).

    Options for encoding:

    * compression (int or str): ``0`` or "no" for no compression (default),
      ``1`` or "zlib" for Zlib compression (same as zip files and PNG), and
      ``2`` or "bz2" for Bz2 compression (more compact but slower writing).
      Note that some BSDF implementations (e.g. JavaScript) may not support
      compression.
    * use_checksum (bool): whether to include a checksum with binary blobs.
    * float64 (bool): Whether to write floats as 64 bit (default) or 32 bit.

    Options for decoding:

    * load_streaming (bool): if True, and the final object in the structure was
      a stream, will make it available as a stream in the decoded object.
    * lazy_blob (bool): if True, bytes are represented as Blob objects that can
      be used to lazily access the data, and also overwrite the data if the
      file is open in a+ mode.
    """

    def __init__(self, extensions=None, **options):
        self._extensions = {}
        self._extensions_by_cls = {}
        if extensions is None:
            extensions = standard_extensions
        for extension in extensions:
            self.add_extension(extension)
        self._parse_options(**options)

    def _parse_options(self, compression=0, use_checksum=False, float64=True, load_streaming=False, lazy_blob=False):
        if isinstance(compression, string_types):
            m = {'no': 0, 'zlib': 1, 'bz2': 2}
            compression = m.get(compression.lower(), compression)
        if compression not in (0, 1, 2):
            raise TypeError('Compression must be 0, 1, 2, "no", "zlib", or "bz2"')
        self._compression = compression
        self._use_checksum = bool(use_checksum)
        self._float64 = bool(float64)
        self._load_streaming = bool(load_streaming)
        self._lazy_blob = bool(lazy_blob)

    def add_extension(self, extension_class):
        """Add an extension to this serializer instance, which must be
        a subclass of Extension. Can be used as a decorator.
        """
        if not (isinstance(extension_class, type) and issubclass(extension_class, Extension)):
            raise TypeError('add_extension() expects a Extension class.')
        extension = extension_class()
        name = extension.name
        if not isinstance(name, str):
            raise TypeError('Extension name must be str.')
        if len(name) == 0 or len(name) > 250:
            raise NameError('Extension names must be nonempty and shorter than 251 chars.')
        if name in self._extensions:
            logger.warning('BSDF warning: overwriting extension "%s", consider removing first' % name)
        cls = extension.cls
        if not cls:
            clss = []
        elif isinstance(cls, (tuple, list)):
            clss = cls
        else:
            clss = [cls]
        for cls in clss:
            if not isinstance(cls, classtypes):
                raise TypeError('Extension classes must be types.')
        for cls in clss:
            self._extensions_by_cls[cls] = (name, extension.encode)
        self._extensions[name] = extension
        return extension_class

    def remove_extension(self, name):
        """Remove a converted by its unique name."""
        if not isinstance(name, str):
            raise TypeError('Extension name must be str.')
        if name in self._extensions:
            self._extensions.pop(name)
        for cls in list(self._extensions_by_cls.keys()):
            if self._extensions_by_cls[cls][0] == name:
                self._extensions_by_cls.pop(cls)

    def _encode(self, f, value, streams, ext_id):
        """Main encoder function."""
        x = encode_type_id
        if value is None:
            f.write(x(b'v', ext_id))
        elif value is True:
            f.write(x(b'y', ext_id))
        elif value is False:
            f.write(x(b'n', ext_id))
        elif isinstance(value, integer_types):
            if -32768 <= value <= 32767:
                f.write(x(b'h', ext_id) + spack('h', value))
            else:
                f.write(x(b'i', ext_id) + spack('<q', value))
        elif isinstance(value, float):
            if self._float64:
                f.write(x(b'd', ext_id) + spack('<d', value))
            else:
                f.write(x(b'f', ext_id) + spack('<f', value))
        elif isinstance(value, unicode_types):
            bb = value.encode('UTF-8')
            f.write(x(b's', ext_id) + lencode(len(bb)))
            f.write(bb)
        elif isinstance(value, (list, tuple)):
            f.write(x(b'l', ext_id) + lencode(len(value)))
            for v in value:
                self._encode(f, v, streams, None)
        elif isinstance(value, dict):
            f.write(x(b'm', ext_id) + lencode(len(value)))
            for key, v in value.items():
                if PY3:
                    assert key.isidentifier()
                else:
                    assert _isidentifier(key)
                name_b = key.encode('UTF-8')
                f.write(lencode(len(name_b)))
                f.write(name_b)
                self._encode(f, v, streams, None)
        elif isinstance(value, bytes):
            f.write(x(b'b', ext_id))
            blob = Blob(value, compression=self._compression, use_checksum=self._use_checksum)
            blob._to_file(f)
        elif isinstance(value, Blob):
            f.write(x(b'b', ext_id))
            value._to_file(f)
        elif isinstance(value, BaseStream):
            if value.mode != 'w':
                raise ValueError('Cannot serialize a read-mode stream.')
            elif isinstance(value, ListStream):
                f.write(x(b'l', ext_id) + spack('<BQ', 255, 0))
            else:
                raise TypeError('Only ListStream is supported')
            if len(streams) > 0:
                raise ValueError('Can only have one stream per file.')
            streams.append(value)
            value._activate(f, self._encode, self._decode)
        else:
            if ext_id is not None:
                raise ValueError('Extension %s wronfully encodes object to another extension object (though it may encode to a list/dict that contains other extension objects).' % ext_id)
            ex = self._extensions_by_cls.get(value.__class__, None)
            if ex is None:
                for name, c in self._extensions.items():
                    if c.match(self, value):
                        ex = (name, c.encode)
                        break
                else:
                    ex = None
            if ex is not None:
                ext_id2, extension_encode = ex
                self._encode(f, extension_encode(self, value), streams, ext_id2)
            else:
                t = 'Class %r is not a valid base BSDF type, nor is it handled by an extension.'
                raise TypeError(t % value.__class__.__name__)

    def _decode(self, f):
        """Main decoder function."""
        char = f.read(1)
        c = char.lower()
        if not char:
            raise EOFError()
        elif char != c:
            n = strunpack('<B', f.read(1))[0]
            ext_id = f.read(n).decode('UTF-8')
        else:
            ext_id = None
        if c == b'v':
            value = None
        elif c == b'y':
            value = True
        elif c == b'n':
            value = False
        elif c == b'h':
            value = strunpack('<h', f.read(2))[0]
        elif c == b'i':
            value = strunpack('<q', f.read(8))[0]
        elif c == b'f':
            value = strunpack('<f', f.read(4))[0]
        elif c == b'd':
            value = strunpack('<d', f.read(8))[0]
        elif c == b's':
            n_s = strunpack('<B', f.read(1))[0]
            if n_s == 253:
                n_s = strunpack('<Q', f.read(8))[0]
            value = f.read(n_s).decode('UTF-8')
        elif c == b'l':
            n = strunpack('<B', f.read(1))[0]
            if n >= 254:
                closed = n == 254
                n = strunpack('<Q', f.read(8))[0]
                if self._load_streaming:
                    value = ListStream(n if closed else 'r')
                    value._activate(f, self._encode, self._decode)
                elif closed:
                    value = [self._decode(f) for i in range(n)]
                else:
                    value = []
                    try:
                        while True:
                            value.append(self._decode(f))
                    except EOFError:
                        pass
            else:
                if n == 253:
                    n = strunpack('<Q', f.read(8))[0]
                value = [self._decode(f) for i in range(n)]
        elif c == b'm':
            value = dict()
            n = strunpack('<B', f.read(1))[0]
            if n == 253:
                n = strunpack('<Q', f.read(8))[0]
            for i in range(n):
                n_name = strunpack('<B', f.read(1))[0]
                if n_name == 253:
                    n_name = strunpack('<Q', f.read(8))[0]
                assert n_name > 0
                name = f.read(n_name).decode('UTF-8')
                value[name] = self._decode(f)
        elif c == b'b':
            if self._lazy_blob:
                value = Blob((f, True))
            else:
                blob = Blob((f, False))
                value = blob.get_bytes()
        else:
            raise RuntimeError('Parse error %r' % char)
        if ext_id is not None:
            extension = self._extensions.get(ext_id, None)
            if extension is not None:
                value = extension.decode(self, value)
            else:
                logger.warning('BSDF warning: no extension found for %r' % ext_id)
        return value

    def encode(self, ob):
        """Save the given object to bytes."""
        f = BytesIO()
        self.save(f, ob)
        return f.getvalue()

    def save(self, f, ob):
        """Write the given object to the given file object."""
        f.write(b'BSDF')
        f.write(struct.pack('<B', VERSION[0]))
        f.write(struct.pack('<B', VERSION[1]))
        streams = []
        self._encode(f, ob, streams, None)
        if len(streams) > 0:
            stream = streams[0]
            if stream._start_pos != f.tell():
                raise ValueError('The stream object must be the last object to be encoded.')

    def decode(self, bb):
        """Load the data structure that is BSDF-encoded in the given bytes."""
        f = BytesIO(bb)
        return self.load(f)

    def load(self, f):
        """Load a BSDF-encoded object from the given file object."""
        f4 = f.read(4)
        if f4 != b'BSDF':
            raise RuntimeError('This does not look like a BSDF file: %r' % f4)
        major_version = strunpack('<B', f.read(1))[0]
        minor_version = strunpack('<B', f.read(1))[0]
        file_version = '%i.%i' % (major_version, minor_version)
        if major_version != VERSION[0]:
            t = 'Reading file with different major version (%s) from the implementation (%s).'
            raise RuntimeError(t % (__version__, file_version))
        if minor_version > VERSION[1]:
            t = 'BSDF warning: reading file with higher minor version (%s) than the implementation (%s).'
            logger.warning(t % (__version__, file_version))
        return self._decode(f)