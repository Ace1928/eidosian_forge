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
class CompositePlot(BokehPlot):
    """
    CompositePlot is an abstract baseclass for plot types that draw
    render multiple axes. It implements methods to add an overall title
    to such a plot.
    """
    sizing_mode = param.ObjectSelector(default=None, objects=['fixed', 'stretch_width', 'stretch_height', 'stretch_both', 'scale_width', 'scale_height', 'scale_both', None], doc='\n\n        How the component should size itself.\n\n        * "fixed" :\n          Component is not responsive. It will retain its original\n          width and height regardless of any subsequent browser window\n          resize events.\n        * "stretch_width"\n          Component will responsively resize to stretch to the\n          available width, without maintaining any aspect ratio. The\n          height of the component depends on the type of the component\n          and may be fixed or fit to component\'s contents.\n        * "stretch_height"\n          Component will responsively resize to stretch to the\n          available height, without maintaining any aspect ratio. The\n          width of the component depends on the type of the component\n          and may be fixed or fit to component\'s contents.\n        * "stretch_both"\n          Component is completely responsive, independently in width\n          and height, and will occupy all the available horizontal and\n          vertical space, even if this changes the aspect ratio of the\n          component.\n        * "scale_width"\n          Component will responsively resize to stretch to the\n          available width, while maintaining the original or provided\n          aspect ratio.\n        * "scale_height"\n          Component will responsively resize to stretch to the\n          available height, while maintaining the original or provided\n          aspect ratio.\n        * "scale_both"\n          Component will responsively resize to both the available\n          width and height, while maintaining the original or provided\n          aspect ratio.\n    ')
    fontsize = param.Parameter(default={'title': '15pt'}, allow_None=True, doc="\n       Specifies various fontsizes of the displayed text.\n\n       Finer control is available by supplying a dictionary where any\n       unmentioned keys reverts to the default sizes, e.g:\n\n          {'title': '15pt'}")

    def _link_dimensioned_streams(self):
        """
        Should perform any linking required to update titles when dimensioned
        streams change.
        """
        streams = [s for s in self.streams if any((k in self.dimensions for k in s.contents))]
        for s in streams:
            s.add_subscriber(self._stream_update, 1)

    def _stream_update(self, **kwargs):
        contents = [k for s in self.streams for k in s.contents]
        key = tuple((None if d in contents else k for d, k in zip(self.dimensions, self.current_key)))
        key = wrap_tuple_streams(key, self.dimensions, self.streams)
        self._get_title_div(key)

    @property
    def current_handles(self):
        """
        Should return a list of plot objects that have changed and
        should be updated.
        """
        return [self.handles['title']] if 'title' in self.handles else []