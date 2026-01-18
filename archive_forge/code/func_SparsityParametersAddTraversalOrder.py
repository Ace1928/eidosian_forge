import flatbuffers
from flatbuffers.compat import import_numpy
def SparsityParametersAddTraversalOrder(builder, traversalOrder):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(traversalOrder), 0)