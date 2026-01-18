import collections
import datetime
import functools
import os
import subprocess
import sys
import time
import errno
from contextlib import contextmanager
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import __version__ as PIL__version__
from PIL import ImageGrab
def _locateAll_opencv(needleImage, haystackImage, grayscale=None, limit=10000, region=None, step=1, confidence=0.999):
    """
    TODO - rewrite this
        faster but more memory-intensive than pure python
        step 2 skips every other row and column = ~3x faster but prone to miss;
            to compensate, the algorithm automatically reduces the confidence
            threshold by 5% (which helps but will not avoid all misses).
        limitations:
          - OpenCV 3.x & python 3.x not tested
          - RGBA images are treated as RBG (ignores alpha channel)
    """
    if grayscale is None:
        grayscale = GRAYSCALE_DEFAULT
    confidence = float(confidence)
    needleImage = _load_cv2(needleImage, grayscale)
    needleHeight, needleWidth = needleImage.shape[:2]
    haystackImage = _load_cv2(haystackImage, grayscale)
    if region:
        haystackImage = haystackImage[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
    else:
        region = (0, 0)
    if haystackImage.shape[0] < needleImage.shape[0] or haystackImage.shape[1] < needleImage.shape[1]:
        raise ValueError('needle dimension(s) exceed the haystack image or region dimensions')
    if step == 2:
        confidence *= 0.95
        needleImage = needleImage[::step, ::step]
        haystackImage = haystackImage[::step, ::step]
    else:
        step = 1
    result = cv2.matchTemplate(haystackImage, needleImage, cv2.TM_CCOEFF_NORMED)
    match_indices = numpy.arange(result.size)[(result > confidence).flatten()]
    matches = numpy.unravel_index(match_indices[:limit], result.shape)
    if len(matches[0]) == 0:
        if USE_IMAGE_NOT_FOUND_EXCEPTION:
            raise ImageNotFoundException('Could not locate the image (highest confidence = %.3f)' % result.max())
        else:
            return
    matchx = matches[1] * step + region[0]
    matchy = matches[0] * step + region[1]
    for x, y in zip(matchx, matchy):
        yield Box(x, y, needleWidth, needleHeight)