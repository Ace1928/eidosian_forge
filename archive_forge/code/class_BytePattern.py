from winappdbg.textio import HexInput
from winappdbg.util import StaticClass, MemoryAddresses
from winappdbg import win32
import warnings
class BytePattern(Pattern):
    """
    Fixed byte pattern.

    @type pattern: str
    @ivar pattern: Byte string to search for.

    @type length: int
    @ivar length: Length of the byte pattern.
    """

    def __init__(self, pattern):
        """
        @type  pattern: str
        @param pattern: Byte string to search for.
        """
        self.pattern = str(pattern)
        self.length = len(pattern)

    def __len__(self):
        """
        Returns the exact length of the pattern.

        @see: L{Pattern.__len__}
        """
        return self.length

    def find(self, buffer, pos=None):
        return (buffer.find(self.pattern, pos), self.length)