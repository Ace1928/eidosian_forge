from jnius import autoclass, PythonJavaClass, java_method
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Fbo, Callback, Rectangle
from kivy.core.camera import CameraBase
import threading
class CameraAndroid(CameraBase):
    """
    Implementation of CameraBase using Android API
    """
    _update_ev = None

    def __init__(self, **kwargs):
        self._android_camera = None
        self._preview_cb = PreviewCallback(self._on_preview_frame)
        self._buflock = threading.Lock()
        super(CameraAndroid, self).__init__(**kwargs)

    def __del__(self):
        self._release_camera()

    def init_camera(self):
        self._release_camera()
        self._android_camera = Camera.open(self._index)
        params = self._android_camera.getParameters()
        width, height = self._resolution
        params.setPreviewSize(width, height)
        supported_focus_modes = self._android_camera.getParameters().getSupportedFocusModes()
        if supported_focus_modes.contains('continuous-picture'):
            params.setFocusMode('continuous-picture')
        self._android_camera.setParameters(params)
        self.fps = 30.0
        pf = params.getPreviewFormat()
        assert pf == ImageFormat.NV21
        self._bufsize = int(ImageFormat.getBitsPerPixel(pf) / 8.0 * width * height)
        self._camera_texture = Texture(width=width, height=height, target=GL_TEXTURE_EXTERNAL_OES, colorfmt='rgba')
        self._surface_texture = SurfaceTexture(int(self._camera_texture.id))
        self._android_camera.setPreviewTexture(self._surface_texture)
        self._fbo = Fbo(size=self._resolution)
        self._fbo['resolution'] = (float(width), float(height))
        self._fbo.shader.fs = '\n            #extension GL_OES_EGL_image_external : require\n            #ifdef GL_ES\n                precision highp float;\n            #endif\n\n            /* Outputs from the vertex shader */\n            varying vec4 frag_color;\n            varying vec2 tex_coord0;\n\n            /* uniform texture samplers */\n            uniform sampler2D texture0;\n            uniform samplerExternalOES texture1;\n            uniform vec2 resolution;\n\n            void main()\n            {\n                vec2 coord = vec2(tex_coord0.y * (\n                    resolution.y / resolution.x), 1. -tex_coord0.x);\n                gl_FragColor = texture2D(texture1, tex_coord0);\n            }\n        '
        with self._fbo:
            self._texture_cb = Callback(lambda instr: self._camera_texture.bind)
            Rectangle(size=self._resolution)

    def _release_camera(self):
        if self._android_camera is None:
            return
        self.stop()
        self._android_camera.release()
        self._android_camera = None
        self._texture = None
        del self._fbo, self._surface_texture, self._camera_texture

    def _on_preview_frame(self, data, camera):
        with self._buflock:
            if self._buffer is not None:
                self._android_camera.addCallbackBuffer(self._buffer)
            self._buffer = data

    def _refresh_fbo(self):
        self._texture_cb.ask_update()
        self._fbo.draw()

    def start(self):
        super(CameraAndroid, self).start()
        with self._buflock:
            self._buffer = None
        for k in range(2):
            buf = b'\x00' * self._bufsize
            self._android_camera.addCallbackBuffer(buf)
        self._android_camera.setPreviewCallbackWithBuffer(self._preview_cb)
        self._android_camera.startPreview()
        if self._update_ev is not None:
            self._update_ev.cancel()
        self._update_ev = Clock.schedule_interval(self._update, 1 / self.fps)

    def stop(self):
        super(CameraAndroid, self).stop()
        if self._update_ev is not None:
            self._update_ev.cancel()
            self._update_ev = None
        self._android_camera.stopPreview()
        self._android_camera.setPreviewCallbackWithBuffer(None)
        with self._buflock:
            self._buffer = None

    def _update(self, dt):
        self._surface_texture.updateTexImage()
        self._refresh_fbo()
        if self._texture is None:
            self._texture = self._fbo.texture
            self.dispatch('on_load')
        self._copy_to_gpu()

    def _copy_to_gpu(self):
        """
        A dummy placeholder (the image is already in GPU) to be consistent
        with other providers.
        """
        self.dispatch('on_texture')

    def grab_frame(self):
        """
        Grab current frame (thread-safe, minimal overhead)
        """
        with self._buflock:
            if self._buffer is None:
                return None
            buf = self._buffer.tostring()
            return buf

    def decode_frame(self, buf):
        """
        Decode image data from grabbed frame.

        This method depends on OpenCV and NumPy - however it is only used for
        fetching the current frame as a NumPy array, and not required when
        this :class:`CameraAndroid` provider is simply used by a
        :class:`~kivy.uix.camera.Camera` widget.
        """
        import numpy as np
        from cv2 import cvtColor
        w, h = self._resolution
        arr = np.fromstring(buf, 'uint8').reshape((h + h / 2, w))
        arr = cvtColor(arr, 93)
        return arr

    def read_frame(self):
        """
        Grab and decode frame in one call
        """
        return self.decode_frame(self.grab_frame())

    @staticmethod
    def get_camera_count():
        """
        Get the number of available cameras.
        """
        return Camera.getNumberOfCameras()