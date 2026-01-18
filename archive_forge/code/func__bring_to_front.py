from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def _bring_to_front(self, touch):
    if self.auto_bring_to_front and self.parent:
        parent = self.parent
        if parent.children[0] is self:
            return
        parent.remove_widget(self)
        parent.add_widget(self)
        self.dispatch('on_bring_to_front', touch)