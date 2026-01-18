import struct
import builtins
import warnings
from collections import namedtuple
class Aifc_write:
    _file = None

    def __init__(self, f):
        if isinstance(f, str):
            file_object = builtins.open(f, 'wb')
            try:
                self.initfp(file_object)
            except:
                file_object.close()
                raise
            if f.endswith('.aiff'):
                self._aifc = 0
        else:
            self.initfp(f)

    def initfp(self, file):
        self._file = file
        self._version = _AIFC_version
        self._comptype = b'NONE'
        self._compname = b'not compressed'
        self._convert = None
        self._nchannels = 0
        self._sampwidth = 0
        self._framerate = 0
        self._nframes = 0
        self._nframeswritten = 0
        self._datawritten = 0
        self._datalength = 0
        self._markers = []
        self._marklength = 0
        self._aifc = 1

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def aiff(self):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._aifc = 0

    def aifc(self):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._aifc = 1

    def setnchannels(self, nchannels):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if nchannels < 1:
            raise Error('bad # of channels')
        self._nchannels = nchannels

    def getnchannels(self):
        if not self._nchannels:
            raise Error('number of channels not set')
        return self._nchannels

    def setsampwidth(self, sampwidth):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if sampwidth < 1 or sampwidth > 4:
            raise Error('bad sample width')
        self._sampwidth = sampwidth

    def getsampwidth(self):
        if not self._sampwidth:
            raise Error('sample width not set')
        return self._sampwidth

    def setframerate(self, framerate):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if framerate <= 0:
            raise Error('bad frame rate')
        self._framerate = framerate

    def getframerate(self):
        if not self._framerate:
            raise Error('frame rate not set')
        return self._framerate

    def setnframes(self, nframes):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._nframes = nframes

    def getnframes(self):
        return self._nframeswritten

    def setcomptype(self, comptype, compname):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if comptype not in (b'NONE', b'ulaw', b'ULAW', b'alaw', b'ALAW', b'G722', b'sowt', b'SOWT'):
            raise Error('unsupported compression type')
        self._comptype = comptype
        self._compname = compname

    def getcomptype(self):
        return self._comptype

    def getcompname(self):
        return self._compname

    def setparams(self, params):
        nchannels, sampwidth, framerate, nframes, comptype, compname = params
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if comptype not in (b'NONE', b'ulaw', b'ULAW', b'alaw', b'ALAW', b'G722', b'sowt', b'SOWT'):
            raise Error('unsupported compression type')
        self.setnchannels(nchannels)
        self.setsampwidth(sampwidth)
        self.setframerate(framerate)
        self.setnframes(nframes)
        self.setcomptype(comptype, compname)

    def getparams(self):
        if not self._nchannels or not self._sampwidth or (not self._framerate):
            raise Error('not all parameters set')
        return _aifc_params(self._nchannels, self._sampwidth, self._framerate, self._nframes, self._comptype, self._compname)

    def setmark(self, id, pos, name):
        if id <= 0:
            raise Error('marker ID must be > 0')
        if pos < 0:
            raise Error('marker position must be >= 0')
        if not isinstance(name, bytes):
            raise Error('marker name must be bytes')
        for i in range(len(self._markers)):
            if id == self._markers[i][0]:
                self._markers[i] = (id, pos, name)
                return
        self._markers.append((id, pos, name))

    def getmark(self, id):
        for marker in self._markers:
            if id == marker[0]:
                return marker
        raise Error('marker {0!r} does not exist'.format(id))

    def getmarkers(self):
        if len(self._markers) == 0:
            return None
        return self._markers

    def tell(self):
        return self._nframeswritten

    def writeframesraw(self, data):
        if not isinstance(data, (bytes, bytearray)):
            data = memoryview(data).cast('B')
        self._ensure_header_written(len(data))
        nframes = len(data) // (self._sampwidth * self._nchannels)
        if self._convert:
            data = self._convert(data)
        self._file.write(data)
        self._nframeswritten = self._nframeswritten + nframes
        self._datawritten = self._datawritten + len(data)

    def writeframes(self, data):
        self.writeframesraw(data)
        if self._nframeswritten != self._nframes or self._datalength != self._datawritten:
            self._patchheader()

    def close(self):
        if self._file is None:
            return
        try:
            self._ensure_header_written(0)
            if self._datawritten & 1:
                self._file.write(b'\x00')
                self._datawritten = self._datawritten + 1
            self._writemarkers()
            if self._nframeswritten != self._nframes or self._datalength != self._datawritten or self._marklength:
                self._patchheader()
        finally:
            self._convert = None
            f = self._file
            self._file = None
            f.close()

    def _lin2alaw(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            import audioop
        return audioop.lin2alaw(data, 2)

    def _lin2ulaw(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            import audioop
        return audioop.lin2ulaw(data, 2)

    def _lin2adpcm(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            import audioop
        if not hasattr(self, '_adpcmstate'):
            self._adpcmstate = None
        data, self._adpcmstate = audioop.lin2adpcm(data, 2, self._adpcmstate)
        return data

    def _lin2sowt(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            import audioop
        return audioop.byteswap(data, 2)

    def _ensure_header_written(self, datasize):
        if not self._nframeswritten:
            if self._comptype in (b'ULAW', b'ulaw', b'ALAW', b'alaw', b'G722', b'sowt', b'SOWT'):
                if not self._sampwidth:
                    self._sampwidth = 2
                if self._sampwidth != 2:
                    raise Error('sample width must be 2 when compressing with ulaw/ULAW, alaw/ALAW, sowt/SOWT or G7.22 (ADPCM)')
            if not self._nchannels:
                raise Error('# channels not specified')
            if not self._sampwidth:
                raise Error('sample width not specified')
            if not self._framerate:
                raise Error('sampling rate not specified')
            self._write_header(datasize)

    def _init_compression(self):
        if self._comptype == b'G722':
            self._convert = self._lin2adpcm
        elif self._comptype in (b'ulaw', b'ULAW'):
            self._convert = self._lin2ulaw
        elif self._comptype in (b'alaw', b'ALAW'):
            self._convert = self._lin2alaw
        elif self._comptype in (b'sowt', b'SOWT'):
            self._convert = self._lin2sowt

    def _write_header(self, initlength):
        if self._aifc and self._comptype != b'NONE':
            self._init_compression()
        self._file.write(b'FORM')
        if not self._nframes:
            self._nframes = initlength // (self._nchannels * self._sampwidth)
        self._datalength = self._nframes * self._nchannels * self._sampwidth
        if self._datalength & 1:
            self._datalength = self._datalength + 1
        if self._aifc:
            if self._comptype in (b'ulaw', b'ULAW', b'alaw', b'ALAW'):
                self._datalength = self._datalength // 2
                if self._datalength & 1:
                    self._datalength = self._datalength + 1
            elif self._comptype == b'G722':
                self._datalength = (self._datalength + 3) // 4
                if self._datalength & 1:
                    self._datalength = self._datalength + 1
        try:
            self._form_length_pos = self._file.tell()
        except (AttributeError, OSError):
            self._form_length_pos = None
        commlength = self._write_form_length(self._datalength)
        if self._aifc:
            self._file.write(b'AIFC')
            self._file.write(b'FVER')
            _write_ulong(self._file, 4)
            _write_ulong(self._file, self._version)
        else:
            self._file.write(b'AIFF')
        self._file.write(b'COMM')
        _write_ulong(self._file, commlength)
        _write_short(self._file, self._nchannels)
        if self._form_length_pos is not None:
            self._nframes_pos = self._file.tell()
        _write_ulong(self._file, self._nframes)
        if self._comptype in (b'ULAW', b'ulaw', b'ALAW', b'alaw', b'G722'):
            _write_short(self._file, 8)
        else:
            _write_short(self._file, self._sampwidth * 8)
        _write_float(self._file, self._framerate)
        if self._aifc:
            self._file.write(self._comptype)
            _write_string(self._file, self._compname)
        self._file.write(b'SSND')
        if self._form_length_pos is not None:
            self._ssnd_length_pos = self._file.tell()
        _write_ulong(self._file, self._datalength + 8)
        _write_ulong(self._file, 0)
        _write_ulong(self._file, 0)

    def _write_form_length(self, datalength):
        if self._aifc:
            commlength = 18 + 5 + len(self._compname)
            if commlength & 1:
                commlength = commlength + 1
            verslength = 12
        else:
            commlength = 18
            verslength = 0
        _write_ulong(self._file, 4 + verslength + self._marklength + 8 + commlength + 16 + datalength)
        return commlength

    def _patchheader(self):
        curpos = self._file.tell()
        if self._datawritten & 1:
            datalength = self._datawritten + 1
            self._file.write(b'\x00')
        else:
            datalength = self._datawritten
        if datalength == self._datalength and self._nframes == self._nframeswritten and (self._marklength == 0):
            self._file.seek(curpos, 0)
            return
        self._file.seek(self._form_length_pos, 0)
        dummy = self._write_form_length(datalength)
        self._file.seek(self._nframes_pos, 0)
        _write_ulong(self._file, self._nframeswritten)
        self._file.seek(self._ssnd_length_pos, 0)
        _write_ulong(self._file, datalength + 8)
        self._file.seek(curpos, 0)
        self._nframes = self._nframeswritten
        self._datalength = datalength

    def _writemarkers(self):
        if len(self._markers) == 0:
            return
        self._file.write(b'MARK')
        length = 2
        for marker in self._markers:
            id, pos, name = marker
            length = length + len(name) + 1 + 6
            if len(name) & 1 == 0:
                length = length + 1
        _write_ulong(self._file, length)
        self._marklength = length + 8
        _write_short(self._file, len(self._markers))
        for marker in self._markers:
            id, pos, name = marker
            _write_short(self._file, id)
            _write_ulong(self._file, pos)
            _write_string(self._file, name)