import os
import sys
from typing import IO, TYPE_CHECKING, Optional
from wandb.errors import CommError
class Progress:
    """A helper class for displaying progress."""
    ITER_BYTES = 1024 * 1024

    def __init__(self, file: IO[bytes], callback: Optional['ProgressFn']=None) -> None:
        self.file = file
        if callback is None:

            def callback_(new_bytes: int, total_bytes: int) -> None:
                pass
            callback = callback_
        self.callback: ProgressFn = callback
        self.bytes_read = 0
        self.len = os.fstat(file.fileno()).st_size

    def read(self, size=-1):
        """Read bytes and call the callback."""
        bites = self.file.read(size)
        self.bytes_read += len(bites)
        if not bites and self.bytes_read < self.len:
            raise CommError('File {} size shrank from {} to {} while it was being uploaded.'.format(self.file.name, self.len, self.bytes_read))
        self.callback(len(bites), self.bytes_read)
        return bites

    def rewind(self) -> None:
        self.callback(-self.bytes_read, 0)
        self.bytes_read = 0
        self.file.seek(0)

    def __getattr__(self, name):
        """Fallback to the file object for attrs not defined here."""
        if hasattr(self.file, name):
            return getattr(self.file, name)
        else:
            raise AttributeError

    def __iter__(self):
        return self

    def __next__(self):
        bites = self.read(self.ITER_BYTES)
        if len(bites) == 0:
            raise StopIteration
        return bites

    def __len__(self):
        return self.len
    next = __next__