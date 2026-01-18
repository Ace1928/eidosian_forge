import os
import win32api
import win32con
import win32event
import win32service
import win32serviceutil
from cherrypy.process import wspbus, plugins
class ConsoleCtrlHandler(plugins.SimplePlugin):
    """A WSPBus plugin for handling Win32 console events (like Ctrl-C)."""

    def __init__(self, bus):
        self.is_set = False
        plugins.SimplePlugin.__init__(self, bus)

    def start(self):
        if self.is_set:
            self.bus.log('Handler for console events already set.', level=20)
            return
        result = win32api.SetConsoleCtrlHandler(self.handle, 1)
        if result == 0:
            self.bus.log('Could not SetConsoleCtrlHandler (error %r)' % win32api.GetLastError(), level=40)
        else:
            self.bus.log('Set handler for console events.', level=20)
            self.is_set = True

    def stop(self):
        if not self.is_set:
            self.bus.log('Handler for console events already off.', level=20)
            return
        try:
            result = win32api.SetConsoleCtrlHandler(self.handle, 0)
        except ValueError:
            result = 1
        if result == 0:
            self.bus.log('Could not remove SetConsoleCtrlHandler (error %r)' % win32api.GetLastError(), level=40)
        else:
            self.bus.log('Removed handler for console events.', level=20)
            self.is_set = False

    def handle(self, event):
        """Handle console control events (like Ctrl-C)."""
        if event in (win32con.CTRL_C_EVENT, win32con.CTRL_LOGOFF_EVENT, win32con.CTRL_BREAK_EVENT, win32con.CTRL_SHUTDOWN_EVENT, win32con.CTRL_CLOSE_EVENT):
            self.bus.log('Console event %s: shutting down bus' % event)
            try:
                self.stop()
            except ValueError:
                pass
            self.bus.exit()
            return 1
        return 0