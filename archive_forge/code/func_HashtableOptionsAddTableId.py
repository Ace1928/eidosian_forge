import flatbuffers
from flatbuffers.compat import import_numpy
def HashtableOptionsAddTableId(builder, tableId):
    builder.PrependInt32Slot(0, tableId, 0)