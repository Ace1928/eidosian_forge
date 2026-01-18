from kivy import Logger
from kivy.core.clipboard import ClipboardBase
from jnius import autoclass, cast
from android.runnable import run_on_ui_thread
from android import python_act
@run_on_ui_thread
def _initialize_clipboard(self):
    PythonActivity._clipboard = cast('android.app.Activity', PythonActivity.mActivity).getSystemService(Context.CLIPBOARD_SERVICE)