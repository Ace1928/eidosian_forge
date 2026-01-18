import flatbuffers
from flatbuffers.compat import import_numpy
def ConversionMetadataAddOptions(builder, options):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(options), 0)