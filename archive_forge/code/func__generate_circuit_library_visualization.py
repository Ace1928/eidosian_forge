from qiskit import QuantumCircuit
from qiskit.utils import optionals as _optionals
@_optionals.HAS_MATPLOTLIB.require_in_call
def _generate_circuit_library_visualization(circuit: QuantumCircuit):
    import matplotlib.pyplot as plt
    circuit = circuit.decompose()
    ops = circuit.count_ops()
    num_nl = circuit.num_nonlocal_gates()
    _fig, (ax0, ax1) = plt.subplots(2, 1)
    circuit.draw('mpl', ax=ax0)
    ax1.axis('off')
    ax1.grid(visible=None)
    ax1.table([[circuit.name], [circuit.width()], [circuit.depth()], [sum(ops.values())], [num_nl]], rowLabels=['Circuit Name', 'Width', 'Depth', 'Total Gates', 'Non-local Gates'])
    plt.tight_layout()
    plt.show()