from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
def _set_layout_manager(self, value):
    lm = self._layout_manager
    if value is lm:
        return
    if lm is not None:
        self._layout_manager = None
        lm.detach_recycleview()
    if value is None:
        return True
    if not isinstance(value, RecycleLayoutManagerBehavior):
        raise ValueError('Expected object based on RecycleLayoutManagerBehavior, got {}'.format(value.__class__))
    self._layout_manager = value
    value.attach_recycleview(self)
    self.refresh_from_layout()
    return True