from kivy.clock import Clock
from kivy.uix.image import Image
from kivy.core.video import Video as CoreVideo
from kivy.resources import resource_find
from kivy.properties import (BooleanProperty, NumericProperty, ObjectProperty,
def _on_video_frame(self, *largs):
    video = self._video
    if not video:
        return
    self.duration = video.duration
    self.position = video.position
    self.texture = video.texture
    self.canvas.ask_update()