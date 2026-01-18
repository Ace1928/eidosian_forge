from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
Recalculate xAvgCharWidth using metrics from ttFont's 'hmtx' table.

        Set it to 0 if the unlikely event 'hmtx' table is not found.
        