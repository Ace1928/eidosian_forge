from kivy.logger import Logger
from kivy.core.window import WindowBase
from kivy.base import EventLoop, ExceptionManager, stopTouchApp
from kivy.lib.vidcore_lite import bcm, egl
from os import environ
def _create_window(self, w, h):
    dst = bcm.Rect(0, 0, w, h)
    src = bcm.Rect(0, 0, w << 16, h << 16)
    display = egl.bcm_display_open(self._rpi_dispmanx_id)
    update = egl.bcm_update_start(0)
    element = egl.bcm_element_add(update, display, self._rpi_dispmanx_layer, dst, src)
    self.win = egl.NativeWindow(element, w, h)
    egl.bcm_update_submit_sync(update)