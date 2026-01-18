import math
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.axes import Axes
import matplotlib.axis as maxis
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.spines import Spine
class RadialTick(maxis.YTick):
    """
    A radial-axis tick.

    This subclass of `.YTick` provides radial ticks with some small
    modification to their re-positioning such that ticks are rotated based on
    axes limits.  This results in ticks that are correctly perpendicular to
    the spine. Labels are also rotated to be perpendicular to the spine, when
    'auto' rotation is enabled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label1.set_rotation_mode('anchor')
        self.label2.set_rotation_mode('anchor')

    def _determine_anchor(self, mode, angle, start):
        if mode == 'auto':
            if start:
                if -90 <= angle <= 90:
                    return ('left', 'center')
                else:
                    return ('right', 'center')
            elif -90 <= angle <= 90:
                return ('right', 'center')
            else:
                return ('left', 'center')
        elif start:
            if angle < -68.5:
                return ('center', 'top')
            elif angle < -23.5:
                return ('left', 'top')
            elif angle < 22.5:
                return ('left', 'center')
            elif angle < 67.5:
                return ('left', 'bottom')
            elif angle < 112.5:
                return ('center', 'bottom')
            elif angle < 157.5:
                return ('right', 'bottom')
            elif angle < 202.5:
                return ('right', 'center')
            elif angle < 247.5:
                return ('right', 'top')
            else:
                return ('center', 'top')
        elif angle < -68.5:
            return ('center', 'bottom')
        elif angle < -23.5:
            return ('right', 'bottom')
        elif angle < 22.5:
            return ('right', 'center')
        elif angle < 67.5:
            return ('right', 'top')
        elif angle < 112.5:
            return ('center', 'top')
        elif angle < 157.5:
            return ('left', 'top')
        elif angle < 202.5:
            return ('left', 'center')
        elif angle < 247.5:
            return ('left', 'bottom')
        else:
            return ('center', 'bottom')

    def update_position(self, loc):
        super().update_position(loc)
        axes = self.axes
        thetamin = axes.get_thetamin()
        thetamax = axes.get_thetamax()
        direction = axes.get_theta_direction()
        offset_rad = axes.get_theta_offset()
        offset = np.rad2deg(offset_rad)
        full = _is_full_circle_deg(thetamin, thetamax)
        if full:
            angle = (axes.get_rlabel_position() * direction + offset) % 360 - 90
            tick_angle = 0
        else:
            angle = (thetamin * direction + offset) % 360 - 90
            if direction > 0:
                tick_angle = np.deg2rad(angle)
            else:
                tick_angle = np.deg2rad(angle + 180)
        text_angle = (angle + 90) % 180 - 90
        mode, user_angle = self._labelrotation
        if mode == 'auto':
            text_angle += user_angle
        else:
            text_angle = user_angle
        if full:
            ha = self.label1.get_horizontalalignment()
            va = self.label1.get_verticalalignment()
        else:
            ha, va = self._determine_anchor(mode, angle, direction > 0)
        self.label1.set_horizontalalignment(ha)
        self.label1.set_verticalalignment(va)
        self.label1.set_rotation(text_angle)
        marker = self.tick1line.get_marker()
        if marker == mmarkers.TICKLEFT:
            trans = mtransforms.Affine2D().rotate(tick_angle)
        elif marker == '_':
            trans = mtransforms.Affine2D().rotate(tick_angle + np.pi / 2)
        elif marker == mmarkers.TICKRIGHT:
            trans = mtransforms.Affine2D().scale(-1, 1).rotate(tick_angle)
        else:
            trans = self.tick1line._marker._transform
        self.tick1line._marker._transform = trans
        if full:
            self.label2.set_visible(False)
            self.tick2line.set_visible(False)
        angle = (thetamax * direction + offset) % 360 - 90
        if direction > 0:
            tick_angle = np.deg2rad(angle)
        else:
            tick_angle = np.deg2rad(angle + 180)
        text_angle = (angle + 90) % 180 - 90
        mode, user_angle = self._labelrotation
        if mode == 'auto':
            text_angle += user_angle
        else:
            text_angle = user_angle
        ha, va = self._determine_anchor(mode, angle, direction < 0)
        self.label2.set_ha(ha)
        self.label2.set_va(va)
        self.label2.set_rotation(text_angle)
        marker = self.tick2line.get_marker()
        if marker == mmarkers.TICKLEFT:
            trans = mtransforms.Affine2D().rotate(tick_angle)
        elif marker == '_':
            trans = mtransforms.Affine2D().rotate(tick_angle + np.pi / 2)
        elif marker == mmarkers.TICKRIGHT:
            trans = mtransforms.Affine2D().scale(-1, 1).rotate(tick_angle)
        else:
            trans = self.tick2line._marker._transform
        self.tick2line._marker._transform = trans