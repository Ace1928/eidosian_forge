from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
class Panose(object):

    def __init__(self, **kwargs):
        _, names, _ = sstruct.getformat(panoseFormat)
        for name in names:
            setattr(self, name, kwargs.pop(name, 0))
        for k in kwargs:
            raise TypeError(f'Panose() got an unexpected keyword argument {k!r}')

    def toXML(self, writer, ttFont):
        formatstring, names, fixes = sstruct.getformat(panoseFormat)
        for name in names:
            writer.simpletag(name, value=getattr(self, name))
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        setattr(self, name, safeEval(attrs['value']))