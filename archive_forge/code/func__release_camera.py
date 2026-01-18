from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
def _release_camera(self):
    if self._android_camera is None:
        return
    self.stop()
    self._android_camera.release()
    self._android_camera = None
    self._texture = None
    del self._fbo, self._surface_texture, self._camera_texture