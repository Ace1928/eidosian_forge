import wave
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
class WaveDecoder(MediaDecoder):

    def get_file_extensions(self):
        return ('.wav', '.wave', '.riff')

    def decode(self, filename, file, streaming=True):
        if streaming:
            return WaveSource(filename, file)
        else:
            return StaticSource(WaveSource(filename, file))