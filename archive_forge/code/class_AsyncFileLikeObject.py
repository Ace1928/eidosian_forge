from io import IOBase, TextIOWrapper
from typing import Iterable, Any, Union, Type, Optional
class AsyncFileLikeObject(IOBase):

    def __init__(self, data: Iterable[Union[bytes, Any]], base: Type=bytes, **kwargs):
        super().__init__()
        self.base: Type[bytes] = base
        self.chunk = base()
        self.offset = 0
        self.iterator = aiter(data)

    async def up_to_iter(self, size: int) -> Iterable[Union[bytes, Any]]:
        """
        Yield up to size bytes from the iterator.
        """
        while size:
            if self.offset == len(self.chunk):
                try:
                    self.chunk = await anext(self.iterator)
                except StopIteration:
                    break
                else:
                    self.offset = 0
            to_yield = min(size, len(self.chunk) - self.offset)
            self.offset += to_yield
            size -= to_yield
            yield self.chunk[self.offset - to_yield:self.offset]

    def readable(self):
        """
        Return True if the stream can be read from. If False, read() will raise
        """
        return True

    def writable(self):
        """
        Return True if the stream supports writing. If False, write() will raise
        """
        return False

    async def read(self, size: int=-1) -> bytes:
        """
        Read and return up to size bytes. If the argument is omitted, None, or
        """
        b = []
        async for chunk in self.up_to_iter(float('inf') if size is None or size < 0 else size):
            b += chunk
        return self.base().join(b)