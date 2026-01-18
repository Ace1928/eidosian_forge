import flatbuffers
from flatbuffers.compat import import_numpy
def EnvironmentAddModelType(builder, modelType):
    builder.PrependInt32Slot(2, modelType, 0)