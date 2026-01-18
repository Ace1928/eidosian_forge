from Xlib import X
from Xlib.protocol import rq
class DisplayError(Exception):

    def __init__(self, display):
        self.display = display

    def __str__(self):
        return 'Display error "%s"' % self.display