from typing import ClassVar, List
import param
from ..io.resources import CDN_DIST
from ..layout import Column
from ..reactive import ReactiveHTML
from ..widgets.base import CompositeWidget
from ..widgets.icon import ToggleIcon
def _update_value(self, event):
    reaction = event.obj._reaction
    icon_value = event.new
    reactions = self.value.copy()
    if icon_value and reaction not in self.value:
        reactions.append(reaction)
    elif not icon_value and reaction in self.value:
        reactions.remove(reaction)
    self.value = reactions