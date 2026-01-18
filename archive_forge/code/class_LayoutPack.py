from __future__ import annotations
import typing
from dataclasses import asdict, dataclass
from matplotlib.layout_engine import LayoutEngine
from matplotlib.text import Text
@dataclass
class LayoutPack:
    """
    Objects required to compute the layout
    """
    axs: list[Axes]
    figure: Figure
    renderer: RendererBase
    theme: theme
    facet: facet
    axis_title_x: Optional[Text] = None
    axis_title_y: Optional[Text] = None
    legends: Optional[legend_artists] = None
    plot_caption: Optional[Text] = None
    plot_subtitle: Optional[Text] = None
    plot_title: Optional[Text] = None
    strip_text_x: Optional[list[StripText]] = None
    strip_text_y: Optional[list[StripText]] = None