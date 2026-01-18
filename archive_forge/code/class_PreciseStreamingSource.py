import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
class PreciseStreamingSource(StreamingSource):
    """Wrap non-precise sources that may over- or undershoot.

    Purpose of this source is to always return data whose length is equal or
    less than in length, where less hints at definite source exhaustion.

    This source is used by pyglet internally, you probably don't need to
    bother with it.

    This source erases AudioData-contained timestamp/duration information and
    events.
    """

    def __init__(self, source: Source) -> None:
        self._source = source
        self._buffer = bytearray()
        self._exhausted = False
        self._is_player_source = False
        self.audio_format = source.audio_format
        self.video_format = source.video_format
        self.info = source.info
        self._duration = source.duration

    @property
    def is_player_source(self) -> bool:
        return self._source.is_player_source

    @is_player_source.setter
    def is_player_source(self, n: bool) -> None:
        self._source.is_player_source = n

    def is_precise(self) -> bool:
        return True

    def seek(self, timestamp: float) -> None:
        self._buffer.clear()
        self._exhausted = False
        self._source.seek(timestamp)

    def get_audio_data(self, num_bytes: int) -> Optional[AudioData]:
        if self._exhausted:
            return None
        if len(self._buffer) < num_bytes:
            required_bytes = num_bytes - len(self._buffer)
            base_attempt = next_or_equal_power_of_two(max(4096, required_bytes + 16))
            attempts = (base_attempt, base_attempt, base_attempt * 2, base_attempt * 8)
            cur_attempt_idx = 0
            empty_bailout = 4
            while True:
                if cur_attempt_idx + 1 < 4:
                    cur_attempt_idx += 1
                res = self._source.get_audio_data(attempts[cur_attempt_idx])
                if res is None:
                    self._exhausted = True
                elif res.length == 0:
                    empty_bailout -= 1
                    if empty_bailout <= 0:
                        self._exhausted = True
                else:
                    empty_bailout = 4
                    self._buffer += res.data
                if len(self._buffer) >= num_bytes or self._exhausted:
                    break
        res = self._buffer[:num_bytes]
        if not res:
            return None
        del self._buffer[:num_bytes]
        return AudioData(res, len(res))

    def get_next_video_timestamp(self) -> Optional[float]:
        return self._source.get_next_video_timestamp()

    def get_next_video_frame(self) -> Optional['AbstractImage']:
        return self._source.get_next_video_frame()

    def save(self, filename: str, file: Optional[BinaryIO]=None, encoder: Optional['MediaEncoder']=None) -> None:
        self._source.save(filename, file, encoder)

    def delete(self) -> None:
        self._source.delete()
        self._buffer.clear()