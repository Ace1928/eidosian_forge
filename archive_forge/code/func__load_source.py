from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.resources import resource_find
from kivy.properties import (
from kivy.logger import Logger
def _load_source(self, *args):
    source = self.source
    if not source:
        self._clear_core_image()
        return
    if not self.is_uri(source):
        source = resource_find(source)
        if not source:
            Logger.error('AsyncImage: Not found <%s>' % self.source)
            self._clear_core_image()
            return
    self._found_source = source
    self._coreimage = image = Loader.image(source, nocache=self.nocache, mipmap=self.mipmap, anim_delay=self.anim_delay)
    image.bind(on_load=self._on_source_load, on_error=self._on_source_error, on_texture=self._on_tex_change)
    self.texture = image.texture