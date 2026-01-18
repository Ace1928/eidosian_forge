import numpy as np
import param
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from packaging.version import Version
from ...core import Dimension
from ...core.options import abbreviated_exception
from ...util.transform import dim as dim_expr
from ..util import map_colors
from .chart import PointPlot
from .element import ColorbarPlot
from .path import PathPlot
from .util import mpl_version
class Plot3D(ColorbarPlot):
    """
    Plot3D provides a common baseclass for mplot3d based
    plots.
    """
    azimuth = param.Integer(default=-60, bounds=(-180, 180), doc='\n        Azimuth angle in the x,y plane.')
    elevation = param.Integer(default=30, bounds=(0, 180), doc='\n        Elevation angle in the z-axis.')
    distance = param.Integer(default=10, bounds=(7, 15), doc='\n        Distance from the plotted object.')
    disable_axes = param.Boolean(default=False, doc='\n        Disable all axes.')
    bgcolor = param.String(default='white', doc='\n        Background color of the axis.')
    labelled = param.List(default=['x', 'y', 'z'], doc="\n        Whether to plot the 'x', 'y' and 'z' labels.")
    projection = param.ObjectSelector(default='3d', objects=['3d'], doc='\n        The projection of the matplotlib axis.')
    show_grid = param.Boolean(default=True, doc='\n        Whether to draw a grid in the figure.')
    xaxis = param.ObjectSelector(default='fixed', objects=['fixed', None], doc='\n        Whether and where to display the xaxis.')
    yaxis = param.ObjectSelector(default='fixed', objects=['fixed', None], doc='\n        Whether and where to display the yaxis.')
    zaxis = param.ObjectSelector(default='fixed', objects=['fixed', None], doc='\n        Whether and where to display the yaxis.')

    def _finalize_axis(self, key, **kwargs):
        """
        Extends the ElementPlot _finalize_axis method to set appropriate
        labels, and axes options for 3D Plots.
        """
        axis = self.handles['axis']
        self.handles['fig'].set_frameon(False)
        axis.grid(self.show_grid)
        axis.view_init(elev=self.elevation, azim=self.azimuth)
        try:
            axis._dist = self.distance
        except Exception:
            axis.dist = self.distance
        if self.xaxis is None:
            axis.w_xaxis.line.set_lw(0.0)
            axis.w_xaxis.label.set_text('')
        if self.yaxis is None:
            axis.w_yaxis.line.set_lw(0.0)
            axis.w_yaxis.label.set_text('')
        if self.zaxis is None:
            axis.w_zaxis.line.set_lw(0.0)
            axis.w_zaxis.label.set_text('')
        if self.disable_axes:
            axis.set_axis_off()
        if mpl_version <= Version('1.5.9'):
            axis.set_axis_bgcolor(self.bgcolor)
        else:
            axis.set_facecolor(self.bgcolor)
        return super()._finalize_axis(key, **kwargs)

    def _draw_colorbar(self, element=None, dim=None, redraw=True):
        if element is None:
            element = self.hmap.last
        artist = self.handles.get('artist', None)
        fig = self.handles['fig']
        ax = self.handles['axis']
        if isinstance(dim, dim_expr):
            dim = dim.dimension
        if dim is None:
            if hasattr(self, 'color_index'):
                dim = element.get_dimension(self.color_index)
            else:
                dim = element.get_dimension(2)
        elif not isinstance(dim, Dimension):
            dim = element.get_dimension(dim)
        label = dim.pprint_label
        cbar = fig.colorbar(artist, shrink=0.7, ax=ax)
        self.handles['cbar'] = cbar
        self.handles['cax'] = cbar.ax
        self._adjust_cbar(cbar, label, dim)