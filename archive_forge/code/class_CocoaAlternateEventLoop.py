import signal
from pyglet import app
from pyglet.app.base import PlatformEventLoop, EventLoop
from pyglet.libs.darwin import cocoapy, AutoReleasePool, ObjCSubclass, PyObjectEncoding, ObjCInstance, send_super, \
class CocoaAlternateEventLoop(EventLoop):
    """This is an alternate loop developed mainly for ARM64 variants of macOS.
    nextEventMatchingMask_untilDate_inMode_dequeue_ is very broken with ctypes calls. Events eventually stop
    working properly after X returns. This event loop differs in that it uses the built-in NSApplication event
    loop. We tie our schedule into it via timer.
    """

    def __init__(self):
        super().__init__()
        self.platform_event_loop = None

    def run(self, interval=1 / 60):
        if not interval:
            self.clock.schedule(self._redraw_windows)
        else:
            self.clock.schedule_interval(self._redraw_windows, interval)
        self.has_exit = False
        from pyglet.window import Window
        Window._enable_event_queue = False
        for window in app.windows:
            window.switch_to()
            window.dispatch_pending_events()
        self.platform_event_loop = app.platform_event_loop
        self.dispatch_event('on_enter')
        self.is_running = True
        self.platform_event_loop.nsapp_start(interval)

    def exit(self):
        """Safely exit the event loop at the end of the current iteration.

        This method is a thread-safe equivalent for setting
        :py:attr:`has_exit` to ``True``.  All waiting threads will be
        interrupted (see :py:meth:`sleep`).
        """
        self.has_exit = True
        if self.platform_event_loop is not None:
            self.platform_event_loop.notify()
        self.is_running = False
        self.dispatch_event('on_exit')
        if self.platform_event_loop is not None:
            self.platform_event_loop.nsapp_stop()