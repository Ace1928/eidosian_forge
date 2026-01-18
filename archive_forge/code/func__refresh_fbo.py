from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
def _refresh_fbo(self):
    self._texture_cb.ask_update()
    self._fbo.draw()