import os
import datetime
from PIL import Image as PILImage
import numpy as np
from traits.api import (
def _get_scan_width(self):
    return self.scan_size[0]