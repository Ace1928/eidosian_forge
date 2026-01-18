import os
import datetime
from PIL import Image as PILImage
import numpy as np
from traits.api import (
def _set_scan_width(self, value):
    self.scan_size = (value, self.scan_size[1])