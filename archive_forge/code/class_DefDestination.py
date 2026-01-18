from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
class DefDestination(HotLink):
    defined = 0

    def link(self, rect, canvas):
        destinationname = self.url
        if not self.defined:
            [x, y, x1, y1] = rect
            canvas.bookmarkHorizontal(destinationname, x, y1)
            self.defined = 1