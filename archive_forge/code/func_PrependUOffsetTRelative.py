from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def PrependUOffsetTRelative(self, off):
    """Prepends an unsigned offset into vector data, relative to where it
        will be written.
        """
    self.Prep(N.UOffsetTFlags.bytewidth, 0)
    if not off <= self.Offset():
        msg = 'flatbuffers: Offset arithmetic error.'
        raise OffsetArithmeticError(msg)
    off2 = self.Offset() - off + N.UOffsetTFlags.bytewidth
    self.PlaceUOffsetT(off2)