import os
from collections import namedtuple
from bisect import bisect_right
from ..construct.lib.container import Container
from ..common.exceptions import DWARFError
from ..common.utils import (struct_parse, dwarf_assert,
from .structs import DWARFStructs
from .compileunit import CompileUnit
from .abbrevtable import AbbrevTable
from .lineprogram import LineProgram
from .callframe import CallFrameInfo
from .locationlists import LocationLists, LocationListsPair
from .ranges import RangeLists, RangeListsPair
from .aranges import ARanges
from .namelut import NameLUT
from .dwarf_util import _get_base_offset
def EH_CFI_entries(self):
    """ Get a list of eh_frame CFI entries from the .eh_frame section.
        """
    cfi = CallFrameInfo(stream=self.eh_frame_sec.stream, size=self.eh_frame_sec.size, address=self.eh_frame_sec.address, base_structs=self.structs, for_eh_frame=True)
    return cfi.get_entries()