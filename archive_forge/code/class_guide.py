from __future__ import annotations
from abc import ABC
from dataclasses import asdict, dataclass, field
from functools import cached_property
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from .._utils import ensure_xy_location, get_opposite_side
from .._utils.registry import Register
from ..themes.theme import theme as Theme
@dataclass
class guide(ABC, metaclass=Register):
    """
    Base class for all guides

    Notes
    -----
    At the moment not all parameters have been fully implemented.
    """
    title: Optional[str] = None
    '\n    Title of the guide. Default is the name of the aesthetic or the\n    name specified using [](`~plotnine.components.labels.lab`)\n    '
    theme: Theme = field(default_factory=Theme)
    'A theme to style the guide. If `None`, the plots theme is used.'
    position: Optional[LegendPosition] = None
    'Where to place the guide relative to the panels.'
    direction: Optional[Orientation] = None
    '\n    Direction of the guide. The default is depends on\n    [](`~plotnine.themes.themeable.legend_position`).\n    '
    reverse: bool = False
    'Whether to reverse the order of the legend keys.'
    order: int = 0
    'Order of this guide among multiple guides.'
    available_aes: set[str] = field(init=False, default_factory=set)

    def __post_init__(self):
        self.hash: str
        self.key: pd.DataFrame
        self.plot_layers: Layers
        self.plot_mapping: aes
        self._elements_cls = GuideElements
        self.elements = cast(GuideElements, None)
        self.guides_elements: GuidesElements

    def legend_aesthetics(self, layer):
        """
        Return the aesthetics that contribute to the legend

        Parameters
        ----------
        layer : Layer
            Layer whose legend is to be drawn

        Returns
        -------
        matched : list
            List of the names of the aethetics that contribute
            to the legend.
        """
        l = layer
        legend_ae = set(self.key.columns) - {'label'}
        all_ae = l.mapping.keys() | (self.plot_mapping if l.inherit_aes else set()) | l.stat.DEFAULT_AES.keys()
        geom_ae = l.geom.REQUIRED_AES | l.geom.DEFAULT_AES.keys()
        matched = all_ae & geom_ae & legend_ae
        matched = list(matched - set(l.geom.aes_params))
        return matched

    def setup(self, guides: guides):
        """
        Setup guide for drawing process
        """
        self.theme = guides.plot.theme + self.theme
        self.theme.setup(guides.plot)
        self.plot_layers = guides.plot.layers
        self.plot_mapping = guides.plot.mapping
        self.elements = self._elements_cls(self.theme, self)
        self.guides_elements = guides.elements

    @property
    def _resolved_position_justification(self) -> tuple[SidePosition, float] | tuple[TupleFloat2, TupleFloat2]:
        """
        Return the final position & justification to draw the guide
        """
        pos = self.elements.position
        just_view = asdict(self.guides_elements.justification)
        if isinstance(pos, str):
            just = cast(float, just_view[pos])
            return (pos, just)
        else:
            if (just := just_view['inside']) is None:
                just = pos
            just = cast(tuple[float, float], just)
            return (pos, just)

    def train(self, scale: scale, aesthetic: Optional[str]=None) -> Self | None:
        """
        Create the key for the guide

        Returns guide if training is successful
        """

    def draw(self) -> PackerBase:
        """
        Draw guide
        """
        raise NotImplementedError

    def create_geoms(self) -> Optional[Self]:
        """
        Create layers of geoms for the guide

        Returns
        -------
        :
            self if geom layers were create or None of no geom layers
            were created.
        """
        raise NotImplementedError