import os
import datetime
from PIL import Image as PILImage
import numpy as np
from traits.api import (
def _scan_size_default(self):
    return (1e-05, 1e-05)