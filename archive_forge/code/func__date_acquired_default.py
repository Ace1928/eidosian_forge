import os
import datetime
from PIL import Image as PILImage
import numpy as np
from traits.api import (
def _date_acquired_default(self):
    return datetime.date.today()