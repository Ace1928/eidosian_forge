import collections
import itertools
from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
class CurvedArrowText(mpl.text.Text):

    def __init__(self, arrow, *args, label_pos=0.5, labels_horizontal=False, ax=None, **kwargs):
        self.arrow = arrow
        self.label_pos = label_pos
        self.labels_horizontal = labels_horizontal
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.x, self.y, self.angle = self._update_text_pos_angle(arrow)
        super().__init__(self.x, self.y, *args, rotation=self.angle, **kwargs)
        self.ax.add_artist(self)

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

    def _update_text_pos_angle(self, arrow):
        path_disp = self._get_arrow_path_disp(arrow)
        (x1, y1), (cx, cy), (x2, y2) = path_disp.vertices
        t = self.label_pos
        tt = 1 - t
        x = tt ** 2 * x1 + 2 * t * tt * cx + t ** 2 * x2
        y = tt ** 2 * y1 + 2 * t * tt * cy + t ** 2 * y2
        if self.labels_horizontal:
            angle = 0
        else:
            change_x = 2 * tt * (cx - x1) + 2 * t * (x2 - cx)
            change_y = 2 * tt * (cy - y1) + 2 * t * (y2 - cy)
            angle = np.arctan2(change_y, change_x) / (2 * np.pi) * 360
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
        x, y = self.ax.transData.inverted().transform((x, y))
        return (x, y, angle)

    def draw(self, renderer):
        self.x, self.y, self.angle = self._update_text_pos_angle(self.arrow)
        self.set_position((self.x, self.y))
        self.set_rotation(self.angle)
        super().draw(renderer)