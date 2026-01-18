from kivy import Logger
from kivy.core.clipboard import ClipboardBase
from jnius import autoclass, cast
from android.runnable import run_on_ui_thread
from android import python_act
def _get_clipboard(f):

    def called(*args, **kargs):
        self = args[0]
        if not PythonActivity._clipboard:
            self._initialize_clipboard()
            import time
            while not PythonActivity._clipboard:
                time.sleep(0.01)
        return f(*args, **kargs)
    return called