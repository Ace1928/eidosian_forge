from hyperframe.exceptions import InvalidFrameError
from hyperframe.frame import (
from .exceptions import (
def _validate_frame_length(self, length):
    """
        Confirm that the frame is an appropriate length.
        """
    if length > self.max_frame_size:
        raise FrameTooLargeError('Received overlong frame: length %d, max %d' % (length, self.max_frame_size))