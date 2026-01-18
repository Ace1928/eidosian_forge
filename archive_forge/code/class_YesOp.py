import numpy as np
import cirq
class YesOp(EmptyOp):

    @property
    def gate(self):
        return Yes()