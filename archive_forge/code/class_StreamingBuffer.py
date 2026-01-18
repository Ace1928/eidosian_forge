from collections import deque
from apitools.base.py import gzip
class StreamingBuffer(object):
    """Provides a file-like object that writes to a temporary buffer.

    When data is read from the buffer, it is permanently removed. This is
    useful when there are memory constraints preventing the entire buffer from
    being stored in memory.
    """

    def __init__(self):
        self.__buf = deque()
        self.__size = 0

    def __len__(self):
        return self.__size

    def __nonzero__(self):
        return bool(self.__size)

    @property
    def length(self):
        return self.__size

    def write(self, data):
        if data is not None and data:
            self.__buf.append(data)
            self.__size += len(data)

    def read(self, size=None):
        """Read at most size bytes from this buffer.

        Bytes read from this buffer are consumed and are permanently removed.

        Args:
          size: If provided, read no more than size bytes from the buffer.
            Otherwise, this reads the entire buffer.

        Returns:
          The bytes read from this buffer.
        """
        if size is None:
            size = self.__size
        ret_list = []
        while size > 0 and self.__buf:
            data = self.__buf.popleft()
            size -= len(data)
            ret_list.append(data)
        if size < 0:
            ret_list[-1], remainder = (ret_list[-1][:size], ret_list[-1][size:])
            self.__buf.appendleft(remainder)
        ret = b''.join(ret_list)
        self.__size -= len(ret)
        return ret