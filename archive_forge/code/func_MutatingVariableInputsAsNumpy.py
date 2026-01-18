import flatbuffers
from flatbuffers.compat import import_numpy
def MutatingVariableInputsAsNumpy(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
    if o != 0:
        return self._tab.GetVectorAsNumpy(flatbuffers.number_types.BoolFlags, o)
    return 0