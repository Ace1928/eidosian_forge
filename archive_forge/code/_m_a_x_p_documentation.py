from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable
Recalculate the font bounding box, and most other maxp values except
        for the TT instructions values. Also recalculate the value of bit 1
        of the flags field and the font bounding box of the 'head' table.
        