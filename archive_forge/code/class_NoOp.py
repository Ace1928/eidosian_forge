import numpy as np
import cirq
class NoOp(EmptyOp):

    @property
    def gate(self):
        return No()