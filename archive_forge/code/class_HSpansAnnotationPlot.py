import itertools
from collections import defaultdict
from html import escape
import numpy as np
import pandas as pd
import param
from bokeh.models import Arrow, BoxAnnotation, NormalHead, Slope, Span, TeeHead
from bokeh.transform import dodge
from panel.models import HTML
from ...core.util import datetime_types, dimension_sanitizer
from ...element import HLine, HLines, HSpans, VLine, VLines, VSpan, VSpans
from ..plot import GenericElementPlot
from .element import AnnotationPlot, ColorbarPlot, CompositeElementPlot, ElementPlot
from .plot import BokehPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
from .util import bokeh32, date_to_integer
class HSpansAnnotationPlot(_SyntheticAnnotationPlot):
    _methods = ('hstrip', 'vstrip')
    _element_default = (HSpans, VSpans)
    style_opts = [*fill_properties, *line_properties, 'level', 'visible']