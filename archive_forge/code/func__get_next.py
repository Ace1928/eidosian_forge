from __future__ import annotations
from io import BytesIO
from . import Image, ImageFile
def _get_next(self):
    ret = self._decoder.get_next()
    self.__physical_frame += 1
    if ret is None:
        self._reset()
        self.seek(0)
        msg = 'failed to decode next frame in WebP file'
        raise EOFError(msg)
    data, timestamp = ret
    duration = timestamp - self.__timestamp
    self.__timestamp = timestamp
    timestamp -= duration
    return (data, timestamp, duration)