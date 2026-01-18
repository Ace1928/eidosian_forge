from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import cached_property
from typing import TYPE_CHECKING, Literal, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import ensure_xy_location
from .._utils.registry import Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import (
from ..mapping.aes import rename_aesthetics
from .guide import guide
@dataclass
class guides:
    """
    Guides for each scale

    Used to assign or remove a particular guide to the scale
    of an aesthetic.
    """
    alpha: Optional[LegendOrColorbar | NoGuide] = None
    'Guide for alpha scale.'
    color: Optional[LegendOrColorbar | NoGuide] = None
    'Guide for color scale.'
    fill: Optional[LegendOrColorbar | NoGuide] = None
    'Guide for fill scale.'
    linetype: Optional[LegendOnly | NoGuide] = None
    'Guide for linetype scale.'
    shape: Optional[LegendOnly | NoGuide] = None
    'Guide for shape scale.'
    size: Optional[LegendOnly | NoGuide] = None
    'Guide for size scale.'
    stroke: Optional[LegendOnly | NoGuide] = None
    'Guide for stroke scale.'
    colour: Optional[LegendOnly | NoGuide] = None
    'Guide for colour scale.'

    def __post_init__(self):
        self.plot: ggplot
        self.plot_scales: Scales
        self.plot_labels: labels_view
        self.elements: GuidesElements
        self._lookup: dict[tuple[scale, ScaledAestheticsName], guide] = {}
        if self.colour is not None and self.color is not None:
            raise ValueError('Got a guide for color and colour, choose one.')
        rename_aesthetics(self)

    def __radd__(self, plot: ggplot):
        """
        Add guides to the plot

        Parameters
        ----------
        plot :
            ggplot object being created

        Returns
        -------
        :
            ggplot object with guides.
        """
        for f in fields(self):
            if (g := getattr(self, f.name)) is not None:
                setattr(plot.guides, f.name, g)
        return plot

    def _build(self) -> Sequence[guide]:
        """
        Build the guides

        Returns
        -------
        :
            The individual guides for which the geoms that draw them have
            have been created.
        """
        return self._create_geoms(self._merge(self._train()))

    def _setup(self, plot: ggplot):
        """
        Setup all guides that will be active
        """
        self.plot = plot
        self.elements = GuidesElements(self.plot.theme)
        guide_lookup = {f.name: g for f in fields(self) if (g := getattr(self, f.name)) is not None}
        for scale in self.plot.scales:
            for ae in scale.aesthetics:
                if (g := guide_lookup.get(ae)) in ('none', False) or (g is None and (g := scale.guide) is None):
                    continue
                if isinstance(g, str):
                    g = Registry[f'guide_{g}']()
                elif not isinstance(g, guide):
                    raise PlotnineError(f'Unknown guide: {g}')
                g.setup(self)
                self._lookup[scale, ae] = g

    def _train(self) -> Sequence[guide]:
        """
        Compute all the required guides

        Returns
        -------
        gdefs : list
            Guides for the plots
        """
        gdefs: list[guide] = []
        for (scale, ae), g in self._lookup.items():
            if not g.elements.position:
                continue
            if 'any' not in g.available_aes and scale.aesthetics[0] not in g.available_aes:
                raise PlotnineError(f'{g.__class__.__name__} cannot be used for {scale.aesthetics}')
            if g.title is None:
                if scale.name:
                    g.title = scale.name
                else:
                    g.title = getattr(self.plot.labels, ae)
                    if g.title is None:
                        warn(f'Cannot generate legend for the {ae!r} aesthetic. Make sure you have mapped a variable to it', PlotnineWarning)
            g = g.train(scale, ae)
            if g is not None:
                gdefs.append(g)
        return gdefs

    def _merge(self, gdefs: Sequence[guide]) -> Sequence[guide]:
        """
        Merge overlapped guides

        For example:

        ```python
         from plotnine import *
         p = (
            ggplot(mtcars, aes(y="wt", x="mpg", colour="factor(cyl)"))
            + stat_smooth(aes(fill="factor(cyl)"), method="lm")
            + geom_point()
         )
        ```

        would create two guides with the same hash
        """
        if not gdefs:
            return []
        definitions = pd.DataFrame({'gdef': gdefs, 'hash': [g.hash for g in gdefs]})
        grouped = definitions.groupby('hash', sort=False)
        gdefs = []
        for name, group in grouped:
            gdef = group['gdef'].iloc[0]
            for g in group['gdef'].iloc[1:]:
                gdef = gdef.merge(g)
            gdefs.append(gdef)
        return gdefs

    def _create_geoms(self, gdefs: Sequence[guide]) -> Sequence[guide]:
        """
        Add geoms to the guide definitions
        """
        return [_g for g in gdefs if (_g := g.create_geoms())]

    def _apply_guide_themes(self, gdefs: list[guide]):
        """
        Apply the theme for each guide
        """
        for g in gdefs:
            g.theme.apply()

    def _assemble_guides(self, gdefs: list[guide], boxes: list[PackerBase]) -> legend_artists:
        """
        Assemble guides into Anchored Offset boxes depending on location
        """
        from matplotlib.font_manager import FontProperties
        from matplotlib.offsetbox import HPacker, VPacker
        from .._mpl.offsetbox import FlexibleAnchoredOffsetbox
        elements = self.elements
        lookup: dict[Orientation, type[PackerBase]] = {'horizontal': HPacker, 'vertical': VPacker}

        def _anchored_offset_box(boxes: list[PackerBase]):
            """
            Put a group of guides into a single box for drawing
            """
            packer = lookup[elements.box]
            box = packer(children=boxes, align=elements.box_just, pad=elements.box_margin, sep=elements.spacing)
            return FlexibleAnchoredOffsetbox(xy_loc=(0.5, 0.5), child=box, pad=1, frameon=False, prop=FontProperties(size=0, stretch=0), bbox_to_anchor=(0, 0), bbox_transform=self.plot.figure.transFigure, borderpad=0.0, zorder=99.1)
        groups: dict[tuple[SidePosition, float] | tuple[TupleFloat2, TupleFloat2], list[PackerBase]] = defaultdict(list)
        for g, b in zip(gdefs, boxes):
            groups[g._resolved_position_justification].append(b)
        legends = legend_artists()
        for (position, just), group in groups.items():
            aob = _anchored_offset_box(group)
            if isinstance(position, str) and isinstance(just, (float, int)):
                setattr(legends, position, outside_legend(aob, just))
            else:
                position = cast(tuple[float, float], position)
                just = cast(tuple[float, float], just)
                legends.inside.append(inside_legend(aob, just, position))
        return legends

    def draw(self) -> Optional[OffsetBox]:
        """
        Draw guides onto the figure

        Returns
        -------
        :matplotlib.offsetbox.Offsetbox | None
            A box that contains all the guides for the plot.
            If there are no guides, **None** is returned.
        """
        if self.elements.position == 'none':
            return
        if not (gdefs := self._build()):
            return
        default = max((g.order for g in gdefs)) + 1
        orders = [default if g.order == 0 else g.order for g in gdefs]
        idx: list[int] = list(np.argsort(orders))
        gdefs = [gdefs[i] for i in idx]
        guide_boxes = [g.draw() for g in gdefs]
        self._apply_guide_themes(gdefs)
        legends = self._assemble_guides(gdefs, guide_boxes)
        for aob in legends.boxes:
            self.plot.figure.add_artist(aob)
        self.plot.theme.targets.legends = legends