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
def _update_datasource(self, source, data):
    """
        Update datasource with data for a new frame.
        """
    if not self.document:
        return
    data = self._postprocess_data(data)
    empty = all((len(v) == 0 for v in data.values()))
    if self.streaming and self.streaming[0].data is self.current_frame.data and self._stream_data and (not empty):
        stream = self.streaming[0]
        if stream._triggering:
            data = {k: v[-stream._chunk_length:] for k, v in data.items()}
            source.stream(data, stream.length)
        return
    if cds_column_replace(source, data):
        source.data = data
    else:
        source.data.update(data)
    if hasattr(self, 'selected') and self.selected is not None:
        self._update_selected(source)