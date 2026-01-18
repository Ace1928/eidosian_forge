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
def _link_dimensioned_streams(self):
    """
        Should perform any linking required to update titles when dimensioned
        streams change.
        """
    streams = [s for s in self.streams if any((k in self.dimensions for k in s.contents))]
    for s in streams:
        s.add_subscriber(self._stream_update, 1)