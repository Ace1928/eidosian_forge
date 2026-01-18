import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
class WindowsDefault(BaseBrowser):

    def open(self, url, new=0, autoraise=True):
        sys.audit('webbrowser.open', url)
        try:
            os.startfile(url)
        except OSError:
            return False
        else:
            return True