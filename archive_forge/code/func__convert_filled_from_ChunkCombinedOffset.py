from __future__ import annotations
from typing import TYPE_CHECKING, cast
import numpy as np
from contourpy._contourpy import FillType, LineType
import contourpy.array as arr
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.typecheck import check_filled, check_lines
from contourpy.types import MOVETO, offset_dtype
def _convert_filled_from_ChunkCombinedOffset(filled: cpy.FillReturn_ChunkCombinedOffset, fill_type_to: FillType) -> cpy.FillReturn:
    if fill_type_to == FillType.ChunkCombinedCode:
        chunk_codes: list[cpy.CodeArray | None] = []
        for points, offsets in zip(*filled):
            if points is None:
                chunk_codes.append(None)
            else:
                if TYPE_CHECKING:
                    assert offsets is not None
                chunk_codes.append(arr.codes_from_offsets_and_points(offsets, points))
        return (filled[0], chunk_codes)
    elif fill_type_to == FillType.ChunkCombinedOffset:
        return filled
    else:
        raise ValueError(f'Conversion from {FillType.ChunkCombinedOffset} to {fill_type_to} not supported')