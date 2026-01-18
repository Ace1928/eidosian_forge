from __future__ import annotations
from collections.abc import Sequence
import io
from typing import TYPE_CHECKING, Any, cast
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
from contourpy import FillType, LineType
from contourpy.convert import convert_filled, convert_lines
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.util.mpl_util import filled_to_mpl_paths, lines_to_mpl_paths
from contourpy.util.renderer import Renderer
class MplDebugRenderer(MplRenderer):
    """Debug renderer implemented using Matplotlib.

    Extends ``MplRenderer`` to add extra information to help in debugging such as markers, arrows,
    text, etc.
    """

    def __init__(self, nrows: int=1, ncols: int=1, figsize: tuple[float, float]=(9, 9), show_frame: bool=True) -> None:
        super().__init__(nrows, ncols, figsize, show_frame)

    def _arrow(self, ax: Axes, line_start: cpy.CoordinateArray, line_end: cpy.CoordinateArray, color: str, alpha: float, arrow_size: float) -> None:
        mid = 0.5 * (line_start + line_end)
        along = line_end - line_start
        along /= np.sqrt(np.dot(along, along))
        right = np.asarray((along[1], -along[0]))
        arrow = np.stack((mid - (along * 0.5 - right) * arrow_size, mid + along * 0.5 * arrow_size, mid - (along * 0.5 + right) * arrow_size))
        ax.plot(arrow[:, 0], arrow[:, 1], '-', c=color, alpha=alpha)

    def filled(self, filled: cpy.FillReturn, fill_type: FillType | str, ax: Axes | int=0, color: str='C1', alpha: float=0.7, line_color: str='C0', line_alpha: float=0.7, point_color: str='C0', start_point_color: str='red', arrow_size: float=0.1) -> None:
        fill_type = as_fill_type(fill_type)
        super().filled(filled, fill_type, ax, color, alpha)
        if line_color is None and point_color is None:
            return
        ax = self._get_ax(ax)
        filled = convert_filled(filled, fill_type, FillType.ChunkCombinedOffset)
        if line_color is not None:
            for points, offsets in zip(*filled):
                if points is None:
                    continue
                for start, end in zip(offsets[:-1], offsets[1:]):
                    xys = points[start:end]
                    ax.plot(xys[:, 0], xys[:, 1], c=line_color, alpha=line_alpha)
                    if arrow_size > 0.0:
                        n = len(xys)
                        for i in range(n - 1):
                            self._arrow(ax, xys[i], xys[i + 1], line_color, line_alpha, arrow_size)
        if point_color is not None:
            for points, offsets in zip(*filled):
                if points is None:
                    continue
                mask = np.ones(offsets[-1], dtype=bool)
                mask[offsets[1:] - 1] = False
                if start_point_color is not None:
                    start_indices = offsets[:-1]
                    mask[start_indices] = False
                ax.plot(points[:, 0][mask], points[:, 1][mask], 'o', c=point_color, alpha=line_alpha)
                if start_point_color is not None:
                    ax.plot(points[:, 0][start_indices], points[:, 1][start_indices], 'o', c=start_point_color, alpha=line_alpha)

    def lines(self, lines: cpy.LineReturn, line_type: LineType | str, ax: Axes | int=0, color: str='C0', alpha: float=1.0, linewidth: float=1, point_color: str='C0', start_point_color: str='red', arrow_size: float=0.1) -> None:
        line_type = as_line_type(line_type)
        super().lines(lines, line_type, ax, color, alpha, linewidth)
        if arrow_size == 0.0 and point_color is None:
            return
        ax = self._get_ax(ax)
        separate_lines = convert_lines(lines, line_type, LineType.Separate)
        if TYPE_CHECKING:
            separate_lines = cast(cpy.LineReturn_Separate, separate_lines)
        if arrow_size > 0.0:
            for line in separate_lines:
                for i in range(len(line) - 1):
                    self._arrow(ax, line[i], line[i + 1], color, alpha, arrow_size)
        if point_color is not None:
            for line in separate_lines:
                start_index = 0
                end_index = len(line)
                if start_point_color is not None:
                    ax.plot(line[0, 0], line[0, 1], 'o', c=start_point_color, alpha=alpha)
                    start_index = 1
                    if line[0][0] == line[-1][0] and line[0][1] == line[-1][1]:
                        end_index -= 1
                ax.plot(line[start_index:end_index, 0], line[start_index:end_index, 1], 'o', c=color, alpha=alpha)

    def point_numbers(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, ax: Axes | int=0, color: str='red') -> None:
        ax = self._get_ax(ax)
        x, y = self._grid_as_2d(x, y)
        z = np.asarray(z)
        ny, nx = z.shape
        for j in range(ny):
            for i in range(nx):
                quad = i + j * nx
                ax.text(x[j, i], y[j, i], str(quad), ha='right', va='top', color=color, clip_on=True)

    def quad_numbers(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, ax: Axes | int=0, color: str='blue') -> None:
        ax = self._get_ax(ax)
        x, y = self._grid_as_2d(x, y)
        z = np.asarray(z)
        ny, nx = z.shape
        for j in range(1, ny):
            for i in range(1, nx):
                quad = i + j * nx
                xmid = x[j - 1:j + 1, i - 1:i + 1].mean()
                ymid = y[j - 1:j + 1, i - 1:i + 1].mean()
                ax.text(xmid, ymid, str(quad), ha='center', va='center', color=color, clip_on=True)

    def z_levels(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, lower_level: float, upper_level: float | None=None, ax: Axes | int=0, color: str='green') -> None:
        ax = self._get_ax(ax)
        x, y = self._grid_as_2d(x, y)
        z = np.asarray(z)
        ny, nx = z.shape
        for j in range(ny):
            for i in range(nx):
                zz = z[j, i]
                if upper_level is not None and zz > upper_level:
                    z_level = 2
                elif zz > lower_level:
                    z_level = 1
                else:
                    z_level = 0
                ax.text(x[j, i], y[j, i], str(z_level), ha='left', va='bottom', color=color, clip_on=True)