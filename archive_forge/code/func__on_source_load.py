from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.resources import resource_find
from kivy.properties import (
from kivy.logger import Logger
def _on_source_load(self, value):
    image = self._coreimage.image
    if not image:
        return
    self.texture = image.texture
    self.dispatch('on_load')