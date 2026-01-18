from Xlib import X
from Xlib.protocol import rq
class DisplayConnectionError(DisplayError):

    def __init__(self, display, msg):
        self.display = display
        self.msg = msg

    def __str__(self):
        return 'Can\'t connect to display "%s": %s' % (self.display, self.msg)