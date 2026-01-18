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
@param.depends('show_legends', 'legend_position', watch=True, on_init=True)
def _update_show_legend(self):
    if self.show_legends is not None:
        select_legends(self.layout, self.show_legends, self.legend_position)