from . import DefaultTable
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
import struct
def _getOffsets(self):
    """
        Calculate offsets to VDMX_Group records.
        For each ratRange return a list of offset values from the beginning of
        the VDMX table to a VDMX_Group.
        """
    lenHeader = sstruct.calcsize(VDMX_HeaderFmt)
    lenRatRange = sstruct.calcsize(VDMX_RatRangeFmt)
    lenOffset = struct.calcsize('>H')
    lenGroupHeader = sstruct.calcsize(VDMX_GroupFmt)
    lenVTable = sstruct.calcsize(VDMX_vTableFmt)
    pos = lenHeader + self.numRatios * lenRatRange + self.numRatios * lenOffset
    groupOffsets = []
    for group in self.groups:
        groupOffsets.append(pos)
        lenGroup = lenGroupHeader + len(group) * lenVTable
        pos += lenGroup
    offsets = []
    for ratio in self.ratRanges:
        groupIndex = ratio['groupIndex']
        offsets.append(groupOffsets[groupIndex])
    return offsets