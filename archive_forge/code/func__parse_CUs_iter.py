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
def _parse_CUs_iter(self, offset=0):
    """ Iterate CU objects in order of appearance in the debug_info section.

            offset:
                The offset of the first CU to yield.  Additional iterations
                will return the sequential unit objects.

            See .iter_CUs(), .get_CU_containing(), and .get_CU_at().
        """
    if self.debug_info_sec is None:
        return
    while offset < self.debug_info_sec.size:
        cu = self._cached_CU_at_offset(offset)
        offset = offset + cu['unit_length'] + cu.structs.initial_length_field_size()
        yield cu