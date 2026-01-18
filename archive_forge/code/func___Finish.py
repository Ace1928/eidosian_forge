from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def __Finish(self, rootTable, sizePrefix, file_identifier=None):
    """Finish finalizes a buffer, pointing to the given `rootTable`."""
    N.enforce_number(rootTable, N.UOffsetTFlags)
    prepSize = N.UOffsetTFlags.bytewidth
    if file_identifier is not None:
        prepSize += N.Int32Flags.bytewidth
    if sizePrefix:
        prepSize += N.Int32Flags.bytewidth
    self.Prep(self.minalign, prepSize)
    if file_identifier is not None:
        self.Prep(N.UOffsetTFlags.bytewidth, encode.FILE_IDENTIFIER_LENGTH)
        file_identifier = N.struct.unpack('>BBBB', file_identifier)
        for i in range(encode.FILE_IDENTIFIER_LENGTH - 1, -1, -1):
            self.Place(file_identifier[i], N.Uint8Flags)
    self.PrependUOffsetTRelative(rootTable)
    if sizePrefix:
        size = len(self.Bytes) - self.Head()
        N.enforce_number(size, N.Int32Flags)
        self.PrependInt32(size)
    self.finished = True
    return self.Head()