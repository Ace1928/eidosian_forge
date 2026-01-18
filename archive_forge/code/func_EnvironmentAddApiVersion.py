import flatbuffers
from flatbuffers.compat import import_numpy
def EnvironmentAddApiVersion(builder, apiVersion):
    builder.PrependUint32Slot(1, apiVersion, 0)