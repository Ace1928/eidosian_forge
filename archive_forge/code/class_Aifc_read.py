import struct
import builtins
import warnings
from collections import namedtuple
class Aifc_read:
    _file = None

    def initfp(self, file):
        self._version = 0
        self._convert = None
        self._markers = []
        self._soundpos = 0
        self._file = file
        chunk = Chunk(file)
        if chunk.getname() != b'FORM':
            raise Error('file does not start with FORM id')
        formdata = chunk.read(4)
        if formdata == b'AIFF':
            self._aifc = 0
        elif formdata == b'AIFC':
            self._aifc = 1
        else:
            raise Error('not an AIFF or AIFF-C file')
        self._comm_chunk_read = 0
        self._ssnd_chunk = None
        while 1:
            self._ssnd_seek_needed = 1
            try:
                chunk = Chunk(self._file)
            except EOFError:
                break
            chunkname = chunk.getname()
            if chunkname == b'COMM':
                self._read_comm_chunk(chunk)
                self._comm_chunk_read = 1
            elif chunkname == b'SSND':
                self._ssnd_chunk = chunk
                dummy = chunk.read(8)
                self._ssnd_seek_needed = 0
            elif chunkname == b'FVER':
                self._version = _read_ulong(chunk)
            elif chunkname == b'MARK':
                self._readmark(chunk)
            chunk.skip()
        if not self._comm_chunk_read or not self._ssnd_chunk:
            raise Error('COMM chunk and/or SSND chunk missing')

    def __init__(self, f):
        if isinstance(f, str):
            file_object = builtins.open(f, 'rb')
            try:
                self.initfp(file_object)
            except:
                file_object.close()
                raise
        else:
            self.initfp(f)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def getfp(self):
        return self._file

    def rewind(self):
        self._ssnd_seek_needed = 1
        self._soundpos = 0

    def close(self):
        file = self._file
        if file is not None:
            self._file = None
            file.close()

    def tell(self):
        return self._soundpos

    def getnchannels(self):
        return self._nchannels

    def getnframes(self):
        return self._nframes

    def getsampwidth(self):
        return self._sampwidth

    def getframerate(self):
        return self._framerate

    def getcomptype(self):
        return self._comptype

    def getcompname(self):
        return self._compname

    def getparams(self):
        return _aifc_params(self.getnchannels(), self.getsampwidth(), self.getframerate(), self.getnframes(), self.getcomptype(), self.getcompname())

    def getmarkers(self):
        if len(self._markers) == 0:
            return None
        return self._markers

    def getmark(self, id):
        for marker in self._markers:
            if id == marker[0]:
                return marker
        raise Error('marker {0!r} does not exist'.format(id))

    def setpos(self, pos):
        if pos < 0 or pos > self._nframes:
            raise Error('position not in range')
        self._soundpos = pos
        self._ssnd_seek_needed = 1

    def readframes(self, nframes):
        if self._ssnd_seek_needed:
            self._ssnd_chunk.seek(0)
            dummy = self._ssnd_chunk.read(8)
            pos = self._soundpos * self._framesize
            if pos:
                self._ssnd_chunk.seek(pos + 8)
            self._ssnd_seek_needed = 0
        if nframes == 0:
            return b''
        data = self._ssnd_chunk.read(nframes * self._framesize)
        if self._convert and data:
            data = self._convert(data)
        self._soundpos = self._soundpos + len(data) // (self._nchannels * self._sampwidth)
        return data

    def _alaw2lin(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            import audioop
        return audioop.alaw2lin(data, 2)

    def _ulaw2lin(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            import audioop
        return audioop.ulaw2lin(data, 2)

    def _adpcm2lin(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            import audioop
        if not hasattr(self, '_adpcmstate'):
            self._adpcmstate = None
        data, self._adpcmstate = audioop.adpcm2lin(data, 2, self._adpcmstate)
        return data

    def _sowt2lin(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            import audioop
        return audioop.byteswap(data, 2)

    def _read_comm_chunk(self, chunk):
        self._nchannels = _read_short(chunk)
        self._nframes = _read_long(chunk)
        self._sampwidth = (_read_short(chunk) + 7) // 8
        self._framerate = int(_read_float(chunk))
        if self._sampwidth <= 0:
            raise Error('bad sample width')
        if self._nchannels <= 0:
            raise Error('bad # of channels')
        self._framesize = self._nchannels * self._sampwidth
        if self._aifc:
            kludge = 0
            if chunk.chunksize == 18:
                kludge = 1
                warnings.warn('Warning: bad COMM chunk size')
                chunk.chunksize = 23
            self._comptype = chunk.read(4)
            if kludge:
                length = ord(chunk.file.read(1))
                if length & 1 == 0:
                    length = length + 1
                chunk.chunksize = chunk.chunksize + length
                chunk.file.seek(-1, 1)
            self._compname = _read_string(chunk)
            if self._comptype != b'NONE':
                if self._comptype == b'G722':
                    self._convert = self._adpcm2lin
                elif self._comptype in (b'ulaw', b'ULAW'):
                    self._convert = self._ulaw2lin
                elif self._comptype in (b'alaw', b'ALAW'):
                    self._convert = self._alaw2lin
                elif self._comptype in (b'sowt', b'SOWT'):
                    self._convert = self._sowt2lin
                else:
                    raise Error('unsupported compression type')
                self._sampwidth = 2
        else:
            self._comptype = b'NONE'
            self._compname = b'not compressed'

    def _readmark(self, chunk):
        nmarkers = _read_short(chunk)
        try:
            for i in range(nmarkers):
                id = _read_short(chunk)
                pos = _read_long(chunk)
                name = _read_string(chunk)
                if pos or name:
                    self._markers.append((id, pos, name))
        except EOFError:
            w = 'Warning: MARK chunk contains only %s marker%s instead of %s' % (len(self._markers), '' if len(self._markers) == 1 else 's', nmarkers)
            warnings.warn(w)