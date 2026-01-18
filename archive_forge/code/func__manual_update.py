from __future__ import annotations
from collections import defaultdict
from typing import (
import param
from bokeh.models import Spacer as BkSpacer, TabPanel as BkTabPanel
from ..models.tabs import Tabs as BkTabs
from ..viewable import Layoutable
from .base import NamedListPanel
def _manual_update(self, events, model, doc, root, parent, comm):
    for event in events:
        if event.name == 'closable':
            for child in model.tabs:
                child.closable = event.new