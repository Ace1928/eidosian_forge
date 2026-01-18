import os
import datetime
from PIL import Image as PILImage
import numpy as np
from traits.api import (
from traitsui.api import Item, View
def _get_pixel_area(self):
    if self.image.size > 0:
        return self.scan_height * self.scan_width / self.image.size
    else:
        return 0