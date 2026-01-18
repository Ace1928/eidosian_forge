import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
@validate('subitems')
def _validate_subitems(self, proposal):
    """Validate subitems list.

        Makes sure only one instance of any given subitem can exist in the
        subitem list.
        """
    subitem_ids = [subitem.model_id for subitem in proposal.value]
    if len(set(subitem_ids)) != len(subitem_ids):
        raise Exception('duplicate subitem detected, only use each subitem once')
    return proposal.value