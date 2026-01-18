from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement
def _all_params(self, context):
    if self.params:
        if callable(self.params):
            params = self.params(context)
        else:
            params = self.params
        if not isinstance(params, list):
            params = [params]
        return params
    else:
        return None