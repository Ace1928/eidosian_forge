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
def _patch_datamodel_ref(cls, props, ref):
    """
        Ensure all DataModels have reference to the root model to ensure
        that they can be cleaned up correctly.
        """
    if isinstance(props, dict):
        for v in props.values():
            cls._patch_datamodel_ref(v, ref)
    elif isinstance(props, list):
        for v in props:
            cls._patch_datamodel_ref(v, ref)
    elif isinstance(props, DataModel):
        props.tags.append(f'__ref:{ref}')