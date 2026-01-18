from typing import ClassVar, List
import param
from ..io.resources import CDN_DIST
from ..layout import Column
from ..reactive import ReactiveHTML
from ..widgets.base import CompositeWidget
from ..widgets.icon import ToggleIcon
@param.depends('options', watch=True)
def _render_icons(self):
    self._rendered_icons = {}
    for option, icon in self.options.items():
        active_icon = self.active_icons.get(option, '')
        icon = ToggleIcon(icon=icon, active_icon=active_icon, value=option in self.value, margin=0)
        icon._reaction = option
        icon.param.watch(self._update_value, 'value')
        self._rendered_icons[option] = icon
    self._composite[:] = list(self._rendered_icons.values())