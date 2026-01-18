from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast
from matplotlib._tight_layout import get_subplotspec_list
from ..facets import facet_grid, facet_null, facet_wrap
from .utils import bbox_in_figure_space, tight_bbox_in_figure_space
def _calculate_panel_spacing_facet_wrap(pack: LayoutPack, spaces: LRTBSpaces) -> WHSpaceParts:
    """
    Calculate spacing parts for facet_wrap
    """
    pack.facet = cast(facet_wrap, pack.facet)
    theme = pack.theme
    ncol = pack.facet.ncol
    nrow = pack.facet.nrow
    W, H = theme.getp('figure_size')
    sw = theme.getp('panel_spacing_x')
    sh = theme.getp('panel_spacing_y') * W / H
    strip_align_x = theme.getp('strip_align_x')
    if strip_align_x > -1:
        sh += spaces.t.top_strip_height * (1 + strip_align_x)
    if pack.facet.free['x']:
        sh += max_xlabels_height(pack)
        sh += max_xticks_height(pack)
    if pack.facet.free['y']:
        sw += max_ylabels_width(pack)
        sw += max_yticks_width(pack)
    w = (spaces.right - spaces.left - sw * (ncol - 1)) / ncol
    h = (spaces.top - spaces.bottom - sh * (nrow - 1)) / nrow
    wspace = sw / w
    hspace = sh / h
    return WHSpaceParts(W, H, w, h, sw, sh, wspace, hspace)