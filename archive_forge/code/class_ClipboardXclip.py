from kivy.utils import platform
from kivy.core.clipboard._clipboard_ext import ClipboardExternalBase
class ClipboardXclip(ClipboardExternalBase):

    @staticmethod
    def _clip(inout, selection):
        pipe = {'std' + inout: subprocess.PIPE}
        return subprocess.Popen(['xclip', '-' + inout, '-selection', selection], **pipe)