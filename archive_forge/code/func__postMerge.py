from fontTools import ttLib
import fontTools.merge.base
from fontTools.merge.cmap import (
from fontTools.merge.layout import layoutPreMerge, layoutPostMerge
from fontTools.merge.options import Options
import fontTools.merge.tables
from fontTools.misc.loggingTools import Timer
from functools import reduce
import sys
import logging
def _postMerge(self, font):
    layoutPostMerge(font)
    if 'OS/2' in font:
        font['OS/2'].recalcAvgCharWidth(font)