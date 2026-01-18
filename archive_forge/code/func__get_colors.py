from __future__ import annotations
import itertools
import logging
import warnings
from typing import TYPE_CHECKING
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_angle
from pymatgen.util.string import unicodeify_spacegroup
def _get_colors(self, color_set, alpha, off_color, custom_colors=None):
    """
        Assign colors according to the surface energies of on_wulff facets.

        Returns:
            tuple: color_list, color_proxy, color_proxy_on_wulff, miller_on_wulff,
            e_surf_on_wulff_list
        """
    color_list = [off_color] * len(self.hkl_list)
    color_proxy_on_wulff = []
    miller_on_wulff = []
    e_surf_on_wulff = [(i, e_surf) for i, e_surf in enumerate(self.e_surf_list) if self.on_wulff[i]]
    c_map = plt.get_cmap(color_set)
    e_surf_on_wulff.sort(key=lambda x: x[1], reverse=False)
    e_surf_on_wulff_list = [x[1] for x in e_surf_on_wulff]
    if len(e_surf_on_wulff) > 1:
        cnorm = mpl.colors.Normalize(vmin=min(e_surf_on_wulff_list), vmax=max(e_surf_on_wulff_list))
    else:
        cnorm = mpl.colors.Normalize(vmin=min(e_surf_on_wulff_list) - 0.1, vmax=max(e_surf_on_wulff_list) + 0.1)
    scalar_map = mpl.cm.ScalarMappable(norm=cnorm, cmap=c_map)
    for i, e_surf in e_surf_on_wulff:
        color_list[i] = scalar_map.to_rgba(e_surf, alpha=alpha)
        if tuple(self.miller_list[i]) in custom_colors:
            color_list[i] = custom_colors[tuple(self.miller_list[i])]
        color_proxy_on_wulff.append(plt.Rectangle((2, 2), 1, 1, fc=color_list[i], alpha=alpha))
        miller_on_wulff.append(self.input_miller_fig[i])
    scalar_map.set_array([x[1] for x in e_surf_on_wulff])
    color_proxy = [plt.Rectangle((2, 2), 1, 1, fc=x, alpha=alpha) for x in color_list]
    return (color_list, color_proxy, color_proxy_on_wulff, miller_on_wulff, e_surf_on_wulff_list)