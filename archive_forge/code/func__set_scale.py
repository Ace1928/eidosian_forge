from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def _set_scale(self, scale):
    rescale = scale * 1.0 / self.scale
    self.apply_transform(Matrix().scale(rescale, rescale, rescale), post_multiply=True, anchor=self.to_local(*self.center))