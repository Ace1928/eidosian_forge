import ctypes
import io
from typing import TYPE_CHECKING, BinaryIO, List, Optional, Union
from pyglet.media.exceptions import MediaException, CannotSeekException
from pyglet.util import next_or_equal_power_of_two
class SourceGroup:
    """Group of like sources to allow gapless playback.

    Seamlessly read data from a group of sources to allow for
    gapless playback. All sources must share the same audio format.
    The first source added sets the format.
    """

    def __init__(self) -> None:
        self.audio_format = None
        self.video_format = None
        self.info = None
        self.duration = 0.0
        self._timestamp_offset = 0.0
        self._dequeued_durations = []
        self._sources = []
        self.is_player_source = False

    def is_precise(self) -> bool:
        return False

    def seek(self, time: float) -> None:
        if self._sources:
            self._sources[0].seek(time)

    def add(self, source: Source) -> None:
        self.audio_format = self.audio_format or source.audio_format
        self.info = self.info or source.info
        source = source.get_queue_source()
        assert source.audio_format == self.audio_format, 'Sources must share the same audio format.'
        self._sources.append(source)
        self.duration += source.duration

    def has_next(self) -> bool:
        return len(self._sources) > 1

    def get_queue_source(self) -> 'SourceGroup':
        return self

    def _advance(self) -> None:
        if self._sources:
            self._timestamp_offset += self._sources[0].duration
            self._dequeued_durations.insert(0, self._sources[0].duration)
            old_source = self._sources.pop(0)
            self.duration -= old_source.duration
            if isinstance(old_source, StreamingSource):
                old_source.delete()

    def get_audio_data(self, num_bytes: float, compensation_time=0.0) -> Optional[AudioData]:
        """Get next audio packet.

        :Parameters:
            `num_bytes` : int
                Hint for preferred size of audio packet; may be ignored.

        :rtype: `AudioData`
        :return: Audio data, or None if there is no more data.
        """
        if not self._sources:
            return None
        buffer = b''
        duration = 0.0
        timestamp = 0.0
        while len(buffer) < num_bytes and self._sources:
            audiodata = self._sources[0].get_audio_data(num_bytes)
            if audiodata:
                buffer += audiodata.data
                duration += audiodata.duration
                timestamp += self._timestamp_offset
            else:
                self._advance()
        return AudioData(buffer, len(buffer), timestamp, duration)