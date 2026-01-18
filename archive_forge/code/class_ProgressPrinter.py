from fontTools import ttLib
from fontTools.misc.textTools import safeEval
from fontTools.ttLib.tables.DefaultTable import DefaultTable
import sys
import os
import logging
class ProgressPrinter(object):

    def __init__(self, title, maxval=100):
        print(title)

    def set(self, val, maxval=None):
        pass

    def increment(self, val=1):
        pass

    def setLabel(self, text):
        print(text)