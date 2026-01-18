from __future__ import annotations
from collections import defaultdict
from typing import (
import param
from bokeh.models import Spacer as BkSpacer, TabPanel as BkTabPanel
from ..models.tabs import Tabs as BkTabs
from ..viewable import Layoutable
from .base import NamedListPanel
def _comm_change(self, doc, ref, comm, subpath, attr, old, new):
    if attr in self._changing.get(ref, []):
        self._changing[ref].remove(attr)
        return
    if attr == 'tabs':
        old, new = self._process_close(ref, attr, old, new)
        if new is None:
            return
    super()._comm_change(doc, ref, comm, subpath, attr, old, new)