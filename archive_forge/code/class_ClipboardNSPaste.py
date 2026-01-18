from kivy.core.clipboard import ClipboardBase
from kivy.utils import platform
class ClipboardNSPaste(ClipboardBase):

    def __init__(self):
        super(ClipboardNSPaste, self).__init__()
        self._clipboard = NSPasteboard.generalPasteboard()

    def get(self, mimetype='text/plain'):
        pb = self._clipboard
        data = pb.stringForType_('public.utf8-plain-text')
        if not data:
            return ''
        return data.UTF8String()

    def put(self, data, mimetype='text/plain'):
        pb = self._clipboard
        pb.clearContents()
        utf8 = NSString.alloc().initWithUTF8String_(data)
        pb.setString_forType_(utf8, 'public.utf8-plain-text')

    def get_types(self):
        return list('text/plain')