from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines, patches
from cirq.qis.states import validate_density_matrix
def _plot_element_of_density_matrix(ax, x, y, r, phase, show_rect=False, show_text=False):
    """Plots a single element of a density matrix

    Args:
        x: x coordinate of the cell we are plotting
        y: y coordinate of the cell we are plotting
        r: the amplitude of the qubit in that cell
        phase: phase of the qubit in that cell, in radians
        show_rect: Boolean on if to show the amplitude rectangle, used for diagonal elements
        show_text: Boolean on if to show text labels or not
        ax: The axes to plot on
    """
    _half_cell_size_after_padding = 1 / 1.1 * 0.5
    _rectangle_margin = 0.01
    _image_opacity = 0.8 if not show_text else 0.4
    circle_out = plt.Circle((x + 0.5, y + 0.5), radius=1 * _half_cell_size_after_padding, fill=False, color='#333333')
    circle_in = plt.Circle((x + 0.5, y + 0.5), radius=r * _half_cell_size_after_padding, fill=True, color='IndianRed', alpha=_image_opacity)
    line = lines.Line2D((x + 0.5, x + 0.5 + np.cos(phase) * _half_cell_size_after_padding), (y + 0.5, y + 0.5 + np.sin(phase) * _half_cell_size_after_padding), color='#333333', alpha=_image_opacity)
    ax.add_artist(circle_in)
    ax.add_artist(circle_out)
    ax.add_artist(line)
    if show_rect:
        rect = patches.Rectangle((x + _rectangle_margin, y + _rectangle_margin), 1.0 - 2 * _rectangle_margin, r * (1 - 2 * _rectangle_margin), alpha=0.25)
        ax.add_artist(rect)
    if show_text:
        plt.text(x + 0.5, y + 0.5, f'{np.round(r, decimals=2)}\n{np.round(phase * 180 / np.pi, decimals=2)} deg', horizontalalignment='center', verticalalignment='center')