from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
def _on_preview_frame(self, data, camera):
    with self._buflock:
        if self._buffer is not None:
            self._android_camera.addCallbackBuffer(self._buffer)
        self._buffer = data