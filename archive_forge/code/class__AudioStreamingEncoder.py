import io
from typing import Iterator, List, Optional
import torch
from torch import Tensor
from torio.io._streaming_media_decoder import _get_afilter_desc, StreamingMediaDecoder as StreamReader
from torio.io._streaming_media_encoder import CodecConfig, StreamingMediaEncoder as StreamWriter
class _AudioStreamingEncoder:
    """Given a waveform, encode on-demand and return bytes"""

    def __init__(self, src: Tensor, sample_rate: int, effect: str, muxer: str, encoder: Optional[str], codec_config: Optional[CodecConfig], frames_per_chunk: int):
        self.src = src
        self.buffer = _StreamingIOBuffer()
        self.writer = StreamWriter(self.buffer, format=muxer)
        self.writer.add_audio_stream(num_channels=src.size(1), sample_rate=sample_rate, format=_get_sample_fmt(src.dtype), encoder=encoder, filter_desc=effect, codec_config=codec_config)
        self.writer.open()
        self.fpc = frames_per_chunk
        self.i_iter = 0

    def read(self, n):
        while not self.buffer._buffer and self.i_iter >= 0:
            self.writer.write_audio_chunk(0, self.src[self.i_iter:self.i_iter + self.fpc])
            self.i_iter += self.fpc
            if self.i_iter >= self.src.size(0):
                self.writer.flush()
                self.writer.close()
                self.i_iter = -1
        return self.buffer.pop(n)