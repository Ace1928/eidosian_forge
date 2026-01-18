from os.path import join
import sys
from typing import Optional
from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import WindowBase
from kivy.input.provider import MotionEventProvider
from kivy.input.motionevent import MotionEvent
from kivy.resources import resource_find
from kivy.utils import platform, deprecated
from kivy.compat import unichr
from collections import deque
def _event_filter(self, action, *largs):
    from kivy.app import App
    if action == 'app_terminating':
        EventLoop.quit = True
    elif action == 'app_lowmemory':
        self.dispatch('on_memorywarning')
    elif action == 'app_willenterbackground':
        from kivy.base import stopTouchApp
        app = App.get_running_app()
        if not app:
            Logger.info('WindowSDL: No running App found, pause.')
        elif not app.dispatch('on_pause'):
            if platform == 'android':
                Logger.info('WindowSDL: App stopped, on_pause() returned False.')
                from android import mActivity
                mActivity.finishAndRemoveTask()
            else:
                Logger.info("WindowSDL: App doesn't support pause mode, stop.")
                stopTouchApp()
                return 0
        self._pause_loop = True
    elif action == 'app_didenterforeground':
        if self._pause_loop:
            self._pause_loop = False
            app = App.get_running_app()
            if app:
                app.dispatch('on_resume')
    elif action == 'windowresized':
        self._size = largs
        self._win.resize_window(*self._size)
        EventLoop.idle()
    return 0