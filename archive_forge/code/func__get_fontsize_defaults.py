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
def _get_fontsize_defaults(self):
    theme = self.renderer.theme
    defaults = {'title': get_default(Title, 'text_font_size', theme), 'legend_title': get_default(Legend, 'title_text_font_size', theme), 'legend': get_default(Legend, 'label_text_font_size', theme), 'label': get_default(Axis, 'axis_label_text_font_size', theme), 'ticks': get_default(Axis, 'major_label_text_font_size', theme), 'cticks': get_default(ColorBar, 'major_label_text_font_size', theme), 'clabel': get_default(ColorBar, 'title_text_font_size', theme)}
    processed = dict(defaults)
    for k, v in defaults.items():
        if isinstance(v, dict) and 'value' in v:
            processed[k] = v['value']
    return processed