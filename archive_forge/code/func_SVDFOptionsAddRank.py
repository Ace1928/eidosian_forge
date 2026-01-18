import flatbuffers
from flatbuffers.compat import import_numpy
def SVDFOptionsAddRank(builder, rank):
    builder.PrependInt32Slot(0, rank, 0)