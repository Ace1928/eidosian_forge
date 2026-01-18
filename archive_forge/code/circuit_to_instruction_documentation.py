from qiskit.circuit.parametertable import ParameterTable, ParameterReferences
from qiskit.exceptions import QiskitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
Build an :class:`~.circuit.Instruction` object from a :class:`.QuantumCircuit`.

    The instruction is anonymous (not tied to a named quantum register),
    and so can be inserted into another circuit. The instruction will
    have the same string name as the circuit.

    Args:
        circuit (QuantumCircuit): the input circuit.
        parameter_map (dict): For parameterized circuits, a mapping from
           parameters in the circuit to parameters to be used in the instruction.
           If None, existing circuit parameters will also parameterize the
           instruction.
        equivalence_library (EquivalenceLibrary): Optional equivalence library
           where the converted instruction will be registered.
        label (str): Optional instruction label.

    Raises:
        QiskitError: if parameter_map is not compatible with circuit

    Return:
        qiskit.circuit.Instruction: an instruction equivalent to the action of the
        input circuit. Upon decomposition, this instruction will
        yield the components comprising the original circuit.

    Example:
        .. code-block::

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.converters import circuit_to_instruction

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)
            circuit_to_instruction(circ)
    