import aifc
import audioop
import struct
import sunau
import wave
from .exceptions import DecodeError
from .base import AudioFile
class RawAudioFile(AudioFile):
    """An AIFF, WAV, or Au file that can be read by the Python standard
    library modules ``wave``, ``aifc``, and ``sunau``.
    """

    def __init__(self, filename):
        self._fh = open(filename, 'rb')
        try:
            self._file = aifc.open(self._fh)
        except aifc.Error:
            self._fh.seek(0)
        else:
            self._needs_byteswap = True
            self._check()
            return
        try:
            self._file = wave.open(self._fh)
        except wave.Error:
            self._fh.seek(0)
            pass
        else:
            self._needs_byteswap = False
            self._check()
            return
        try:
            self._file = sunau.open(self._fh)
        except sunau.Error:
            self._fh.seek(0)
            pass
        else:
            self._needs_byteswap = True
            self._check()
            return
        self._fh.close()
        raise UnsupportedError()

    def _check(self):
        """Check that the files' parameters allow us to decode it and
        raise an error otherwise.
        """
        if self._file.getsampwidth() not in SUPPORTED_WIDTHS:
            self.close()
            raise BitWidthError()

    def close(self):
        """Close the underlying file."""
        self._file.close()
        self._fh.close()

    @property
    def channels(self):
        """Number of audio channels."""
        return self._file.getnchannels()

    @property
    def samplerate(self):
        """Sample rate in Hz."""
        return self._file.getframerate()

    @property
    def duration(self):
        """Length of the audio in seconds (a float)."""
        return float(self._file.getnframes()) / self.samplerate

    def read_data(self, block_samples=1024):
        """Generates blocks of PCM data found in the file."""
        old_width = self._file.getsampwidth()
        while True:
            data = self._file.readframes(block_samples)
            if not data:
                break
            data = audioop.lin2lin(data, old_width, TARGET_WIDTH)
            if self._needs_byteswap and self._file.getcomptype() != 'sowt':
                data = byteswap(data)
            yield data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __iter__(self):
        return self.read_data()