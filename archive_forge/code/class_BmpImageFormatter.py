import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
class BmpImageFormatter(ImageFormatter):
    """
    Create a bitmap image from source code. This uses the Python Imaging Library to
    generate a pixmap from the source code.

    .. versionadded:: 1.0
    """
    name = 'img_bmp'
    aliases = ['bmp', 'bitmap']
    filenames = ['*.bmp']
    default_image_format = 'bmp'