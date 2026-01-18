import inspect
import re
import urllib
from typing import List as LList
from typing import Optional, Union
from .... import __version__ as wandb_ver
from .... import termwarn
from ...public import Api as PublicApi
from ._panels import UnknownPanel, WeavePanel, panel_mapping, weave_panels
from .runset import Runset
from .util import (
from .validators import OneOf, TypeValidator
@active_runset.setter
def active_runset(self, name):
    json_path = self._get_path('active_runset')
    index = None
    for i, rs in enumerate(self.runsets):
        if rs.name == name:
            index = i
            break
    nested_set(self, json_path, index)