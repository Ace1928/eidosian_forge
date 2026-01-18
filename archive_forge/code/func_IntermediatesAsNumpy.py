import flatbuffers
from flatbuffers.compat import import_numpy
def IntermediatesAsNumpy(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
    if o != 0:
        return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
    return 0