from . import DefaultTable
import struct
 TSI{0,1,2,3,5} are private tables used by Microsoft Visual TrueType (VTT)
tool to store its hinting source data.

TSI0 is the index table containing the lengths and offsets for the glyph
programs and 'extra' programs ('fpgm', 'prep', and 'cvt') that are contained
in the TSI1 table.
