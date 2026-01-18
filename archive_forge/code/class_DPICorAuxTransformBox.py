from __future__ import annotations
from matplotlib.offsetbox import (
from matplotlib.patches import bbox_artist as mbbox_artist
from matplotlib.transforms import Affine2D, Bbox
from .patches import InsideStrokedRectangle
class DPICorAuxTransformBox(AuxTransformBox):
    """
    DPI Corrected AuxTransformBox
    """

    def __init__(self, aux_transform):
        super().__init__(aux_transform)
        self.dpi_transform = Affine2D()
        self._dpi_corrected = False

    def get_transform(self):
        """
        Return the [](`~matplotlib.transforms.Transform`) applied
        to the children
        """
        return self.aux_transform + self.dpi_transform + self.ref_offset_transform + self.offset_transform

    def _correct_dpi(self, renderer):
        if not self._dpi_corrected:
            dpi_cor = renderer.points_to_pixels(1.0)
            self.dpi_transform.clear()
            self.dpi_transform.scale(dpi_cor, dpi_cor)
            self._dpi_corrected = True

    def get_bbox(self, renderer):
        self._correct_dpi(renderer)
        return super().get_bbox(renderer)

    def draw(self, renderer):
        """
        Draw the children
        """
        self._correct_dpi(renderer)
        for c in self.get_children():
            c.draw(renderer)
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.0))
        self.stale = False