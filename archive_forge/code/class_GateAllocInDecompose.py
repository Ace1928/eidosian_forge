import cirq
class GateAllocInDecompose(cirq.Gate):

    def __init__(self, num_alloc: int=1):
        self.num_alloc = num_alloc

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_with_context_(self, qubits, context):
        assert context is not None
        qm = context.qubit_manager
        for q in qm.qalloc(self.num_alloc):
            yield cirq.CNOT(qubits[0], q)
            qm.qfree([q])

    def __str__(self):
        return 'TestGateAlloc'