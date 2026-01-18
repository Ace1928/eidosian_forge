from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity, Function
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.printing.latex import latex
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .experimental_lambdify import (vectorized_lambdify, lambdify)
from sympy.plotting.textplot import textplot
def _process_series(self, series, ax, parent):
    np = import_module('numpy')
    mpl_toolkits = import_module('mpl_toolkits', import_kwargs={'fromlist': ['mplot3d']})
    xlims, ylims, zlims = ([], [], [])
    for s in series:
        if s.is_2Dline:
            x, y = s.get_data()
            if isinstance(s.line_color, (int, float)) or callable(s.line_color):
                segments = self.get_segments(x, y)
                collection = self.LineCollection(segments)
                collection.set_array(s.get_color_array())
                ax.add_collection(collection)
            else:
                lbl = _str_or_latex(s.label)
                line, = ax.plot(x, y, label=lbl, color=s.line_color)
        elif s.is_contour:
            ax.contour(*s.get_meshes())
        elif s.is_3Dline:
            x, y, z = s.get_data()
            if isinstance(s.line_color, (int, float)) or callable(s.line_color):
                art3d = mpl_toolkits.mplot3d.art3d
                segments = self.get_segments(x, y, z)
                collection = art3d.Line3DCollection(segments)
                collection.set_array(s.get_color_array())
                ax.add_collection(collection)
            else:
                lbl = _str_or_latex(s.label)
                ax.plot(x, y, z, label=lbl, color=s.line_color)
            xlims.append(s._xlim)
            ylims.append(s._ylim)
            zlims.append(s._zlim)
        elif s.is_3Dsurface:
            x, y, z = s.get_meshes()
            collection = ax.plot_surface(x, y, z, cmap=getattr(self.cm, 'viridis', self.cm.jet), rstride=1, cstride=1, linewidth=0.1)
            if isinstance(s.surface_color, (float, int, Callable)):
                color_array = s.get_color_array()
                color_array = color_array.reshape(color_array.size)
                collection.set_array(color_array)
            else:
                collection.set_color(s.surface_color)
            xlims.append(s._xlim)
            ylims.append(s._ylim)
            zlims.append(s._zlim)
        elif s.is_implicit:
            points = s.get_raster()
            if len(points) == 2:
                x, y = _matplotlib_list(points[0])
                ax.fill(x, y, facecolor=s.line_color, edgecolor='None')
            else:
                ListedColormap = self.matplotlib.colors.ListedColormap
                colormap = ListedColormap(['white', s.line_color])
                xarray, yarray, zarray, plot_type = points
                if plot_type == 'contour':
                    ax.contour(xarray, yarray, zarray, cmap=colormap)
                else:
                    ax.contourf(xarray, yarray, zarray, cmap=colormap)
        else:
            raise NotImplementedError('{} is not supported in the SymPy plotting module with matplotlib backend. Please report this issue.'.format(ax))
    Axes3D = mpl_toolkits.mplot3d.Axes3D
    if not isinstance(ax, Axes3D):
        ax.autoscale_view(scalex=ax.get_autoscalex_on(), scaley=ax.get_autoscaley_on())
    else:
        if xlims:
            xlims = np.array(xlims)
            xlim = (np.amin(xlims[:, 0]), np.amax(xlims[:, 1]))
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([0, 1])
        if ylims:
            ylims = np.array(ylims)
            ylim = (np.amin(ylims[:, 0]), np.amax(ylims[:, 1]))
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([0, 1])
        if zlims:
            zlims = np.array(zlims)
            zlim = (np.amin(zlims[:, 0]), np.amax(zlims[:, 1]))
            ax.set_zlim(zlim)
        else:
            ax.set_zlim([0, 1])
    if parent.xscale and (not isinstance(ax, Axes3D)):
        ax.set_xscale(parent.xscale)
    if parent.yscale and (not isinstance(ax, Axes3D)):
        ax.set_yscale(parent.yscale)
    if not isinstance(ax, Axes3D) or self.matplotlib.__version__ >= '1.2.0':
        ax.set_autoscale_on(parent.autoscale)
    if parent.axis_center:
        val = parent.axis_center
        if isinstance(ax, Axes3D):
            pass
        elif val == 'center':
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')
        elif val == 'auto':
            xl, xh = ax.get_xlim()
            yl, yh = ax.get_ylim()
            pos_left = ('data', 0) if xl * xh <= 0 else 'center'
            pos_bottom = ('data', 0) if yl * yh <= 0 else 'center'
            ax.spines['left'].set_position(pos_left)
            ax.spines['bottom'].set_position(pos_bottom)
        else:
            ax.spines['left'].set_position(('data', val[0]))
            ax.spines['bottom'].set_position(('data', val[1]))
    if not parent.axis:
        ax.set_axis_off()
    if parent.legend:
        if ax.legend():
            ax.legend_.set_visible(parent.legend)
    if parent.margin:
        ax.set_xmargin(parent.margin)
        ax.set_ymargin(parent.margin)
    if parent.title:
        ax.set_title(parent.title)
    if parent.xlabel:
        xlbl = _str_or_latex(parent.xlabel)
        ax.set_xlabel(xlbl, position=(1, 0))
    if parent.ylabel:
        ylbl = _str_or_latex(parent.ylabel)
        ax.set_ylabel(ylbl, position=(0, 1))
    if isinstance(ax, Axes3D) and parent.zlabel:
        zlbl = _str_or_latex(parent.zlabel)
        ax.set_zlabel(zlbl, position=(0, 1))
    if parent.annotations:
        for a in parent.annotations:
            ax.annotate(**a)
    if parent.markers:
        for marker in parent.markers:
            m = marker.copy()
            args = m.pop('args')
            ax.plot(*args, **m)
    if parent.rectangles:
        for r in parent.rectangles:
            rect = self.matplotlib.patches.Rectangle(**r)
            ax.add_patch(rect)
    if parent.fill:
        ax.fill_between(**parent.fill)
    if parent.xlim:
        ax.set_xlim(parent.xlim)
    if parent.ylim:
        ax.set_ylim(parent.ylim)