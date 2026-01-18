from __future__ import annotations
import datetime as dt
import difflib
import inspect
import logging
import re
import sys
import textwrap
from collections import Counter, defaultdict, namedtuple
from functools import lru_cache, partial
from pprint import pformat
from typing import (
import numpy as np
import param
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.model import DataModel
from bokeh.models import ImportedStyleSheet
from packaging.version import Version
from param.parameterized import (
from .io.document import unlocked
from .io.model import hold
from .io.notebook import push
from .io.resources import (
from .io.state import set_curdoc, state
from .models.reactive_html import (
from .util import (
from .viewable import Layoutable, Renderable, Viewable
@classmethod
def _loaded(cls) -> bool:
    """
        Whether the component has been loaded.
        """
    return cls._extension_name is None or (cls._extension_name in ReactiveHTMLMetaclass._loaded_extensions and (state._extensions is None or cls._extension_name in state._extensions))