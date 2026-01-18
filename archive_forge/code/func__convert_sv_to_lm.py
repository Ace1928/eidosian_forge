from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
def _convert_sv_to_lm(self, x, y):
    lm = self.layout_manager
    tree = [lm]
    parent = lm.parent
    while parent is not None and parent is not self:
        tree.append(parent)
        parent = parent.parent
    if parent is not self:
        raise Exception('The layout manager must be a sub child of the recycleview. Could not find {} in the parent tree of {}'.format(self, lm))
    for widget in reversed(tree):
        x, y = widget.to_local(x, y)
    return (x, y)