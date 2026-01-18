from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def __clear(self):
    """Clear the Axes."""
    if hasattr(self, 'patch'):
        patch_visible = self.patch.get_visible()
    else:
        patch_visible = True
    xaxis_visible = self.xaxis.get_visible()
    yaxis_visible = self.yaxis.get_visible()
    for axis in self._axis_map.values():
        axis.clear()
    for spine in self.spines.values():
        spine._clear()
    self.ignore_existing_data_limits = True
    self.callbacks = cbook.CallbackRegistry(signals=['xlim_changed', 'ylim_changed', 'zlim_changed'])
    if mpl.rcParams['xtick.minor.visible']:
        self.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    if mpl.rcParams['ytick.minor.visible']:
        self.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    self._xmargin = mpl.rcParams['axes.xmargin']
    self._ymargin = mpl.rcParams['axes.ymargin']
    self._tight = None
    self._use_sticky_edges = True
    self._get_lines = _process_plot_var_args()
    self._get_patches_for_fill = _process_plot_var_args('fill')
    self._gridOn = mpl.rcParams['axes.grid']
    old_children, self._children = (self._children, [])
    for chld in old_children:
        chld.axes = chld.figure = None
    self._mouseover_set = _OrderedSet()
    self.child_axes = []
    self._current_image = None
    self._projection_init = None
    self.legend_ = None
    self.containers = []
    self.grid(False)
    self.grid(self._gridOn, which=mpl.rcParams['axes.grid.which'], axis=mpl.rcParams['axes.grid.axis'])
    props = font_manager.FontProperties(size=mpl.rcParams['axes.titlesize'], weight=mpl.rcParams['axes.titleweight'])
    y = mpl.rcParams['axes.titley']
    if y is None:
        y = 1.0
        self._autotitlepos = True
    else:
        self._autotitlepos = False
    self.title = mtext.Text(x=0.5, y=y, text='', fontproperties=props, verticalalignment='baseline', horizontalalignment='center')
    self._left_title = mtext.Text(x=0.0, y=y, text='', fontproperties=props.copy(), verticalalignment='baseline', horizontalalignment='left')
    self._right_title = mtext.Text(x=1.0, y=y, text='', fontproperties=props.copy(), verticalalignment='baseline', horizontalalignment='right')
    title_offset_points = mpl.rcParams['axes.titlepad']
    self._set_title_offset_trans(title_offset_points)
    for _title in (self.title, self._left_title, self._right_title):
        self._set_artist_props(_title)
    self.patch = self._gen_axes_patch()
    self.patch.set_figure(self.figure)
    self.patch.set_facecolor(self._facecolor)
    self.patch.set_edgecolor('none')
    self.patch.set_linewidth(0)
    self.patch.set_transform(self.transAxes)
    self.set_axis_on()
    self.xaxis.set_clip_path(self.patch)
    self.yaxis.set_clip_path(self.patch)
    if self._sharex is not None:
        self.xaxis.set_visible(xaxis_visible)
        self.patch.set_visible(patch_visible)
    if self._sharey is not None:
        self.yaxis.set_visible(yaxis_visible)
        self.patch.set_visible(patch_visible)
    for name, axis in self._axis_map.items():
        share = getattr(self, f'_share{name}')
        if share is not None:
            getattr(self, f'share{name}')(share)
        else:
            if self.name == 'polar':
                axis._set_scale('linear')
            axis._set_lim(0, 1, auto=True)
    self._update_transScale()
    self.stale = True