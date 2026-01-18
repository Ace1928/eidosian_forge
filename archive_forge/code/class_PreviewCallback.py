from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
class PreviewCallback(PythonJavaClass):
    """
    Interface used to get back the preview frame of the Android Camera
    """
    __javainterfaces__ = ('android.hardware.Camera$PreviewCallback',)

    def __init__(self, callback):
        super(PreviewCallback, self).__init__()
        self._callback = callback

    @java_method('([BLandroid/hardware/Camera;)V')
    def onPreviewFrame(self, data, camera):
        self._callback(data, camera)