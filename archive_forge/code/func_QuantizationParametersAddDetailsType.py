import flatbuffers
from flatbuffers.compat import import_numpy
def QuantizationParametersAddDetailsType(builder, detailsType):
    builder.PrependUint8Slot(4, detailsType, 0)