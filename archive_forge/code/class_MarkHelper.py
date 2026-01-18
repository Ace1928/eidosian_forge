from fontTools import ttLib
from fontTools.ttLib.tables._c_m_a_p import cmap_classes
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import ValueRecord, valueRecordFormatDict
from fontTools.otlLib import builder as otl
from contextlib import contextmanager
from fontTools.ttLib import newTable
from fontTools.feaLib.lookupDebugInfo import LOOKUP_DEBUG_ENV_VAR, LOOKUP_DEBUG_INFO_KEY
from operator import setitem
import os
import logging
class MarkHelper(object):

    def __init__(self):
        for Which in ('Mark', 'Base'):
            for What in ('Coverage', 'Array', 'Count', 'Record', 'Anchor'):
                key = Which + What
                if Which == 'Mark' and What in ('Count', 'Record', 'Anchor'):
                    value = key
                else:
                    value = getattr(self, Which) + What
                if value == 'LigatureRecord':
                    value = 'LigatureAttach'
                setattr(self, key, value)
                if What != 'Count':
                    klass = getattr(ot, value)
                    setattr(self, key + 'Class', klass)