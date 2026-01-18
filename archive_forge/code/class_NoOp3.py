import numpy as np
import cirq
class NoOp3(EmptyOp):

    @property
    def gate(self):
        return No3()