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
def _fontsize(self, key, label='fontsize', common=True):
    """
        Converts integer fontsizes to a string specifying
        fontsize in pt.
        """
    size = super()._fontsize(key, label, common)
    return {k: v if isinstance(v, str) else f'{v}pt' for k, v in size.items()}