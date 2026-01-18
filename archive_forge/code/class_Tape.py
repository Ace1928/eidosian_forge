from tensorflow.python import pywrap_tfe
class Tape(object):
    """Represents a gradient propagation trace."""
    __slots__ = ['_tape']

    def __init__(self, tape):
        self._tape = tape

    def watched_variables(self):
        return pywrap_tfe.TFE_Py_TapeWatchedVariables(self._tape)