import cirq
import cirq.contrib.qcircuit as ccq
import cirq.testing as ct
class MagicGate(cirq.testing.ThreeQubitGate):

    def __str__(self):
        return 'MagicGate'