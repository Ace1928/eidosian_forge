from collections import defaultdict
from itertools import groupby
import numpy as np
import param
from bokeh.layouts import gridplot
from bokeh.models import (
from bokeh.models.layouts import TabPanel, Tabs
from ...core import (
from ...core.options import SkipRendering
from ...core.util import (
from ...selection import NoOpSelectionDisplay
from ..links import Link
from ..plot import (
from ..util import attach_streams, collate, displayable
from .links import LinkCallback
from .util import (
def _get_title_div(self, key, default_fontsize='15pt', width=450):
    title_div = None
    title = self._format_title(key) if self.show_title else ''
    if not title:
        return title_div
    title_json = theme_attr_json(self.renderer.theme, 'Title')
    color = title_json.get('text_color', None)
    font = title_json.get('text_font', 'Arial')
    fontstyle = title_json.get('text_font_style', 'bold')
    fontsize = self._fontsize('title').get('fontsize', default_fontsize)
    if fontsize == default_fontsize:
        fontsize = title_json.get('text_font_size', default_fontsize)
        if 'em' in fontsize:
            fontsize = str(float(fontsize[:-2]) + 0.25) + 'em'
    title_tags = self._title_template.format(color=color, font=font, fontstyle=fontstyle, fontsize=fontsize, title=title)
    if 'title' in self.handles:
        title_div = self.handles['title']
    else:
        title_div = Div(width=width, styles={'white-space': 'nowrap'})
    title_div.text = title_tags
    return title_div