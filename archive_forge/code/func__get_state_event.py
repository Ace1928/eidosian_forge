import os
import win32api
import win32con
import win32event
import win32service
import win32serviceutil
from cherrypy.process import wspbus, plugins
def _get_state_event(self, state):
    """Return a win32event for the given state (creating it if needed)."""
    try:
        return self.events[state]
    except KeyError:
        event = win32event.CreateEvent(None, 0, 0, 'WSPBus %s Event (pid=%r)' % (state.name, os.getpid()))
        self.events[state] = event
        return event