import warnings
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array
from ....stats.density_utils import histogram
from ...plot_utils import _scale_fig_size, color_from_dim, set_xticklabels, vectorized_to_hex
from . import backend_show, create_axes_grid, matplotlib_kwarg_dealiaser
def _make_hover_annotation(fig, ax, sc_plot, coord_labels, rgba_c, hover_format):
    """Show data point label when hovering over it with mouse."""
    annot = ax.annotate('', xy=(0, 0), xytext=(0, 0), textcoords='offset points', bbox=dict(boxstyle='round', fc='w', alpha=0.4), arrowprops=dict(arrowstyle='->'))
    annot.set_visible(False)
    xmid = np.mean(ax.get_xlim())
    ymid = np.mean(ax.get_ylim())
    offset = 10

    def update_annot(ind):
        idx = ind['ind'][0]
        pos = sc_plot.get_offsets()[idx]
        annot_text = hover_format.format(idx, coord_labels[idx])
        annot.xy = pos
        annot.set_position((-offset if pos[0] > xmid else offset, -offset if pos[1] > ymid else offset))
        annot.set_text(annot_text)
        annot.get_bbox_patch().set_facecolor(rgba_c[idx])
        annot.set_ha('right' if pos[0] > xmid else 'left')
        annot.set_va('top' if pos[1] > ymid else 'bottom')

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc_plot.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            elif vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()
    fig.canvas.mpl_connect('motion_notify_event', hover)