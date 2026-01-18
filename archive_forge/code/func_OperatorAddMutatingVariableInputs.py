import flatbuffers
from flatbuffers.compat import import_numpy
def OperatorAddMutatingVariableInputs(builder, mutatingVariableInputs):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(mutatingVariableInputs), 0)