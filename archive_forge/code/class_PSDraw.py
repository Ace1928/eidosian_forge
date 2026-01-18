from __future__ import annotations
import sys
from . import EpsImagePlugin
class PSDraw:
    """
    Sets up printing to the given file. If ``fp`` is omitted,
    ``sys.stdout.buffer`` or ``sys.stdout`` is assumed.
    """

    def __init__(self, fp=None):
        if not fp:
            try:
                fp = sys.stdout.buffer
            except AttributeError:
                fp = sys.stdout
        self.fp = fp

    def begin_document(self, id=None):
        """Set up printing of a document. (Write PostScript DSC header.)"""
        self.fp.write(b'%!PS-Adobe-3.0\nsave\n/showpage { } def\n%%EndComments\n%%BeginDocument\n')
        self.fp.write(EDROFF_PS)
        self.fp.write(VDI_PS)
        self.fp.write(b'%%EndProlog\n')
        self.isofont = {}

    def end_document(self):
        """Ends printing. (Write PostScript DSC footer.)"""
        self.fp.write(b'%%EndDocument\nrestore showpage\n%%End\n')
        if hasattr(self.fp, 'flush'):
            self.fp.flush()

    def setfont(self, font, size):
        """
        Selects which font to use.

        :param font: A PostScript font name
        :param size: Size in points.
        """
        font = bytes(font, 'UTF-8')
        if font not in self.isofont:
            self.fp.write(b'/PSDraw-%s ISOLatin1Encoding /%s E\n' % (font, font))
            self.isofont[font] = 1
        self.fp.write(b'/F0 %d /PSDraw-%s F\n' % (size, font))

    def line(self, xy0, xy1):
        """
        Draws a line between the two points. Coordinates are given in
        PostScript point coordinates (72 points per inch, (0, 0) is the lower
        left corner of the page).
        """
        self.fp.write(b'%d %d %d %d Vl\n' % (*xy0, *xy1))

    def rectangle(self, box):
        """
        Draws a rectangle.

        :param box: A tuple of four integers, specifying left, bottom, width and
           height.
        """
        self.fp.write(b'%d %d M 0 %d %d Vr\n' % box)

    def text(self, xy, text):
        """
        Draws text at the given position. You must use
        :py:meth:`~PIL.PSDraw.PSDraw.setfont` before calling this method.
        """
        text = bytes(text, 'UTF-8')
        text = b'\\('.join(text.split(b'('))
        text = b'\\)'.join(text.split(b')'))
        xy += (text,)
        self.fp.write(b'%d %d M (%s) S\n' % xy)

    def image(self, box, im, dpi=None):
        """Draw a PIL image, centered in the given box."""
        if not dpi:
            if im.mode == '1':
                dpi = 200
            else:
                dpi = 100
        x = im.size[0] * 72 / dpi
        y = im.size[1] * 72 / dpi
        xmax = float(box[2] - box[0])
        ymax = float(box[3] - box[1])
        if x > xmax:
            y = y * xmax / x
            x = xmax
        if y > ymax:
            x = x * ymax / y
            y = ymax
        dx = (xmax - x) / 2 + box[0]
        dy = (ymax - y) / 2 + box[1]
        self.fp.write(b'gsave\n%f %f translate\n' % (dx, dy))
        if (x, y) != im.size:
            sx = x / im.size[0]
            sy = y / im.size[1]
            self.fp.write(b'%f %f scale\n' % (sx, sy))
        EpsImagePlugin._save(im, self.fp, None, 0)
        self.fp.write(b'\ngrestore\n')