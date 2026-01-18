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
def _postprocess_data(self, data):
    """
        Applies necessary type transformation to the data before
        it is set on a ColumnDataSource.
        """
    new_data = {}
    for k, values in data.items():
        values = decode_bytes(values)
        if len(values) and isinstance(values[0], cftime_types):
            if any((v.calendar not in _STANDARD_CALENDARS for v in values)):
                self.param.warning('Converting cftime.datetime from a non-standard calendar (%s) to a standard calendar for plotting. This may lead to subtle errors in formatting dates, for accurate tick formatting switch to the matplotlib backend.' % values[0].calendar)
            values = cftime_to_timestamp(values, 'ms')
        new_data[k] = values
    return new_data