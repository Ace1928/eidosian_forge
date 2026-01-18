import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
class MacOSX(BaseBrowser):
    """Launcher class for Aqua browsers on Mac OS X

        Optionally specify a browser name on instantiation.  Note that this
        will not work for Aqua browsers if the user has moved the application
        package after installation.

        If no browser is specified, the default browser, as specified in the
        Internet System Preferences panel, will be used.
        """

    def __init__(self, name):
        warnings.warn(f'{self.__class__.__name__} is deprecated in 3.11 use MacOSXOSAScript instead.', DeprecationWarning, stacklevel=2)
        self.name = name

    def open(self, url, new=0, autoraise=True):
        sys.audit('webbrowser.open', url)
        assert "'" not in url
        if not ':' in url:
            url = 'file:' + url
        new = int(bool(new))
        if self.name == 'default':
            script = 'open location "%s"' % url.replace('"', '%22')
        else:
            if self.name == 'OmniWeb':
                toWindow = ''
            else:
                toWindow = 'toWindow %d' % (new - 1)
            cmd = 'OpenURL "%s"' % url.replace('"', '%22')
            script = 'tell application "%s"\n                                activate\n                                %s %s\n                            end tell' % (self.name, cmd, toWindow)
        osapipe = os.popen('osascript', 'w')
        if osapipe is None:
            return False
        osapipe.write(script)
        rc = osapipe.close()
        return not rc