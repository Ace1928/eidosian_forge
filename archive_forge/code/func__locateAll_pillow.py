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
def _locateAll_pillow(needleImage, haystackImage, grayscale=None, limit=None, region=None, step=1, confidence=None):
    """
    TODO
    """
    if confidence is not None:
        raise NotImplementedError('The confidence keyword argument is only available if OpenCV is installed.')
    if grayscale is None:
        grayscale = GRAYSCALE_DEFAULT
    needleFileObj = None
    if isinstance(needleImage, str):
        needleFileObj = open(needleImage, 'rb')
        needleImage = Image.open(needleFileObj)
    haystackFileObj = None
    if isinstance(haystackImage, str):
        haystackFileObj = open(haystackImage, 'rb')
        haystackImage = Image.open(haystackFileObj)
    if region is not None:
        haystackImage = haystackImage.crop((region[0], region[1], region[0] + region[2], region[1] + region[3]))
    else:
        region = (0, 0)
    if grayscale:
        needleImage = ImageOps.grayscale(needleImage)
        haystackImage = ImageOps.grayscale(haystackImage)
    else:
        if needleImage.mode == 'RGBA':
            needleImage = needleImage.convert('RGB')
        if haystackImage.mode == 'RGBA':
            haystackImage = haystackImage.convert('RGB')
    needleWidth, needleHeight = needleImage.size
    haystackWidth, haystackHeight = haystackImage.size
    needleImageData = tuple(needleImage.getdata())
    haystackImageData = tuple(haystackImage.getdata())
    needleImageRows = [needleImageData[y * needleWidth:(y + 1) * needleWidth] for y in range(needleHeight)]
    needleImageFirstRow = needleImageRows[0]
    assert len(needleImageFirstRow) == needleWidth, 'The calculated width of first row of the needle image is not the same as the width of the image.'
    assert [len(row) for row in needleImageRows] == [needleWidth] * needleHeight, "The needleImageRows aren't the same size as the original image."
    numMatchesFound = 0
    step = 1
    if step == 1:
        firstFindFunc = _kmp
    else:
        firstFindFunc = _steppingFind
    for y in range(haystackHeight):
        for matchx in firstFindFunc(needleImageFirstRow, haystackImageData[y * haystackWidth:(y + 1) * haystackWidth], step):
            foundMatch = True
            for searchy in range(1, needleHeight, step):
                haystackStart = (searchy + y) * haystackWidth + matchx
                if needleImageData[searchy * needleWidth:(searchy + 1) * needleWidth] != haystackImageData[haystackStart:haystackStart + needleWidth]:
                    foundMatch = False
                    break
            if foundMatch:
                numMatchesFound += 1
                yield Box(matchx + region[0], y + region[1], needleWidth, needleHeight)
                if limit is not None and numMatchesFound >= limit:
                    if needleFileObj is not None:
                        needleFileObj.close()
                    if haystackFileObj is not None:
                        haystackFileObj.close()
                    return
    if needleFileObj is not None:
        needleFileObj.close()
    if haystackFileObj is not None:
        haystackFileObj.close()
    if numMatchesFound == 0:
        if USE_IMAGE_NOT_FOUND_EXCEPTION:
            raise ImageNotFoundException('Could not locate the image.')
        else:
            return