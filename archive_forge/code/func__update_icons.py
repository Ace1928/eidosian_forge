from typing import ClassVar, List
import param
from ..io.resources import CDN_DIST
from ..layout import Column
from ..reactive import ReactiveHTML
from ..widgets.base import CompositeWidget
from ..widgets.icon import ToggleIcon
@param.depends('value', watch=True)
def _update_icons(self):
    for option, icon in self._rendered_icons.items():
        icon.value = option in self.value