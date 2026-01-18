from __future__ import annotations
from typing import TYPE_CHECKING, cast
import numpy as np
from contourpy._contourpy import FillType, LineType
import contourpy.array as arr
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.typecheck import check_filled, check_lines
from contourpy.types import MOVETO, offset_dtype
def _convert_filled_from_OuterOffset(filled: cpy.FillReturn_OuterOffset, fill_type_to: FillType) -> cpy.FillReturn:
    if fill_type_to == FillType.OuterCode:
        separate_codes = [arr.codes_from_offsets(offsets) for offsets in filled[1]]
        return (filled[0], separate_codes)
    elif fill_type_to == FillType.OuterOffset:
        return filled
    if len(filled[0]) > 0:
        points = arr.concat_points(filled[0])
        offsets = arr.concat_offsets(filled[1])
    else:
        points = None
        offsets = None
    if fill_type_to == FillType.ChunkCombinedCode:
        return ([points], [None if offsets is None else arr.codes_from_offsets(offsets)])
    elif fill_type_to == FillType.ChunkCombinedOffset:
        return ([points], [offsets])
    elif fill_type_to == FillType.ChunkCombinedCodeOffset:
        if offsets is None:
            ret1: cpy.FillReturn_ChunkCombinedCodeOffset = ([None], [None], [None])
        else:
            codes = arr.codes_from_offsets(offsets)
            outer_offsets = arr.offsets_from_lengths(filled[0])
            ret1 = ([points], [codes], [outer_offsets])
        return ret1
    elif fill_type_to == FillType.ChunkCombinedOffsetOffset:
        if points is None:
            ret2: cpy.FillReturn_ChunkCombinedOffsetOffset = ([None], [None], [None])
        else:
            outer_offsets = arr.outer_offsets_from_list_of_offsets(filled[1])
            ret2 = ([points], [offsets], [outer_offsets])
        return ret2
    else:
        raise ValueError(f'Invalid FillType {fill_type_to}')