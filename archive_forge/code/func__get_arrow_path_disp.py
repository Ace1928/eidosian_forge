import collections
import itertools
from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def _get_arrow_path_disp(self, arrow):
    """
            This is part of FancyArrowPatch._get_path_in_displaycoord
            It omits the second part of the method where path is converted
                to polygon based on width
            The transform is taken from ax, not the object, as the object
                has not been added yet, and doesn't have transform
            """
    dpi_cor = arrow._dpi_cor
    trans_data = self.ax.transData
    if arrow._posA_posB is not None:
        posA = arrow._convert_xy_units(arrow._posA_posB[0])
        posB = arrow._convert_xy_units(arrow._posA_posB[1])
        posA, posB = trans_data.transform((posA, posB))
        _path = arrow.get_connectionstyle()(posA, posB, patchA=arrow.patchA, patchB=arrow.patchB, shrinkA=arrow.shrinkA * dpi_cor, shrinkB=arrow.shrinkB * dpi_cor)
    else:
        _path = trans_data.transform_path(arrow._path_original)
    return _path