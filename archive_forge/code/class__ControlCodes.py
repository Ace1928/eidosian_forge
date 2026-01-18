import os
import win32api
import win32con
import win32event
import win32service
import win32serviceutil
from cherrypy.process import wspbus, plugins
class _ControlCodes(dict):
    """Control codes used to "signal" a service via ControlService.

    User-defined control codes are in the range 128-255. We generally use
    the standard Python value for the Linux signal and add 128. Example:

        >>> signal.SIGUSR1
        10
        control_codes['graceful'] = 128 + 10
    """

    def key_for(self, obj):
        """For the given value, return its corresponding key."""
        for key, val in self.items():
            if val is obj:
                return key
        raise ValueError('The given object could not be found: %r' % obj)