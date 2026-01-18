from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
def _set_data_model(self, value):
    data_model = self._data_model
    if value is data_model:
        return
    if data_model is not None:
        self._data_model = None
        data_model.detach_recycleview()
    if value is None:
        return True
    if not isinstance(value, RecycleDataModelBehavior):
        raise ValueError('Expected object based on RecycleDataModelBehavior, got {}'.format(value.__class__))
    self._data_model = value
    value.attach_recycleview(self)
    self.refresh_from_data()
    return True