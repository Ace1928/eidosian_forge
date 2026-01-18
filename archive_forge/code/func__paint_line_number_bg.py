import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def _paint_line_number_bg(self, im):
    """
        Paint the line number background on the image.
        """
    if not self.line_numbers:
        return
    if self.line_number_fg is None:
        return
    draw = ImageDraw.Draw(im)
    recth = im.size[-1]
    rectw = self.image_pad + self.line_number_width - self.line_number_pad
    draw.rectangle([(0, 0), (rectw, recth)], fill=self.line_number_bg)
    draw.line([(rectw, 0), (rectw, recth)], fill=self.line_number_fg)
    del draw