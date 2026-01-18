import struct, sys, time, os
import zlib
import builtins
import io
import _compression
class BadGzipFile(OSError):
    """Exception raised in some cases for invalid gzip files."""