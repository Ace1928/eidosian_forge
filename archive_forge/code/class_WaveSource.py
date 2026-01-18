import wave
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
class WaveSource(StreamingSource):

    def __init__(self, filename, file=None):
        if file is None:
            file = open(filename, 'rb')
            self._file = file
        try:
            self._wave = wave.open(file)
        except wave.Error as e:
            raise WAVEDecodeException(e)
        nchannels, sampwidth, framerate, nframes, comptype, compname = self._wave.getparams()
        self.audio_format = AudioFormat(channels=nchannels, sample_size=sampwidth * 8, sample_rate=framerate)
        self._bytes_per_frame = nchannels * sampwidth
        self._duration = nframes / framerate
        self._duration_per_frame = self._duration / nframes
        self._num_frames = nframes
        self._wave.rewind()

    def __del__(self):
        if hasattr(self, '_file'):
            self._file.close()

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        num_frames = max(1, num_bytes // self._bytes_per_frame)
        data = self._wave.readframes(num_frames)
        if not data:
            return None
        timestamp = self._wave.tell() / self.audio_format.sample_rate
        duration = num_frames / self.audio_format.sample_rate
        return AudioData(data, len(data), timestamp, duration, [])

    def seek(self, timestamp):
        timestamp = max(0.0, min(timestamp, self._duration))
        position = int(timestamp / self._duration_per_frame)
        self._wave.setpos(position)