from __future__ import annotations
from typing import List
from dataclasses import dataclass
from qiskit import circuit
from qiskit.circuit.quantumregister import Qubit, QuantumRegister
from qiskit.transpiler.exceptions import LayoutError
from qiskit.converters import isinstanceint
@dataclass
class TranspileLayout:
    """Layout attributes from output circuit from transpiler.

    The transpiler in general is unitary-perserving up to permutations caused
    by setting and applying initial layout during the :ref:`layout_stage`
    and :class:`~.SwapGate` insertion during the :ref:`routing_stage`. To
    provide an interface to reason about these permutations caused by
    the :mod:`~qiskit.transpiler`. In general the normal interface to access
    and reason about the layout transformations made by the transpiler is to
    use the helper methods defined on this class.

    For example, looking at the initial layout, the transpiler can potentially
    remap the order of the qubits in your circuit as it fits the circuit to
    the target backend. If the input circuit was:

    .. plot:
       :include-source:

       from qiskit.circuit import QuantumCircuit, QuantumRegister

       qr = QuantumRegister(3, name="MyReg")
       qc = QuantumCircuit(qr)
       qc.h(0)
       qc.cx(0, 1)
       qc.cx(0, 2)
       qc.draw("mpl")

    Then during the layout stage the transpiler reorders the qubits to be:

    .. plot:
       :include-source:

       from qiskit import QuantumCircuit

       qc = QuantumCircuit(3)
       qc.h(2)
       qc.cx(2, 1)
       qc.cx(2, 0)
       qc.draw("mpl")

    then the output of the :meth:`.initial_virtual_layout` would be
    equivalent to::

        Layout({
            qr[0]: 2,
            qr[1]: 1,
            qr[2]: 0,
        })

    (it is also this attribute in the :meth:`.QuantumCircuit.draw` and
    :func:`.circuit_drawer` which is used to display the mapping of qubits to
    positions in circuit visualizations post-transpilation)

    Building on this above example for final layout, if the transpiler needed to
    insert swap gates during routing so the output circuit became:

    .. plot:
       :include-source:

       from qiskit import QuantumCircuit

       qc = QuantumCircuit(3)
       qc.h(2)
       qc.cx(2, 1)
       qc.swap(0, 1)
       qc.cx(2, 1)
       qc.draw("mpl")

    then the output of the :meth:`routing_permutation` method would be::

        [1, 0, 2]

    which maps the qubits at each position to their final position after any swap
    insertions caused by routing.

    There are three public attributes associated with the class, however these
    are mostly provided for backwards compatibility and represent the internal
    state from the transpiler. They are defined as:

      * :attr:`initial_layout` - This attribute is used to model the
        permutation caused by the :ref:`layout_stage` it contains a
        :class:`~.Layout` object that maps the input :class:`~.QuantumCircuit`\\s
        :class:`~.circuit.Qubit` objects to the position in the output
        :class:`.QuantumCircuit.qubits` list.
      * :attr:`input_qubit_mapping` - This attribute is used to retain
        input ordering of the original :class:`~.QuantumCircuit` object. It
        maps the virtual :class:`~.circuit.Qubit` object from the original circuit
        (and :attr:`initial_layout`) to its corresponding position in
        :attr:`.QuantumCircuit.qubits` in the original circuit. This
        is needed when computing the permutation of the :class:`Operator` of
        the circuit (and used by :meth:`.Operator.from_circuit`).
      * :attr:`final_layout` - This is a :class:`~.Layout` object used to
        model the output permutation caused ny any :class:`~.SwapGate`\\s
        inserted into the :class:`~.QuantumCircuit` during the
        :ref:`routing_stage`. It maps the output circuit's qubits from
        :class:`.QuantumCircuit.qubits` in the output circuit to the final
        position after routing. It is **not** a mapping from the original
        input circuit's position to the final position at the end of the
        transpiled circuit. If you need this you can use the
        :meth:`.final_index_layout` to generate this. If this is set to ``None``
        this indicates that routing was not run and it can be considered
        equivalent to a trivial layout with the qubits from the output circuit's
        :attr:`~.QuantumCircuit.qubits` list.
    """
    initial_layout: Layout
    input_qubit_mapping: dict[circuit.Qubit, int]
    final_layout: Layout | None = None
    _input_qubit_count: int | None = None
    _output_qubit_list: List[Qubit] | None = None

    def initial_virtual_layout(self, filter_ancillas: bool=False) -> Layout:
        """Return a :class:`.Layout` object for the initial layout.

        This returns a mapping of virtual :class:`~.circuit.Qubit` objects in the input
        circuit to the physical qubit selected during layout. This is analogous
        to the :attr:`.initial_layout` attribute.

        Args:
            filter_ancillas: If set to ``True`` only qubits in the input circuit
                will be in the returned layout. Any ancilla qubits added to the
                output circuit will be filtered from the returned object.
        Returns:
            A layout object mapping the input circuit's :class:`~.circuit.Qubit`
            objects to the selected physical qubits.
        """
        if not filter_ancillas:
            return self.initial_layout
        return Layout({k: v for k, v in self.initial_layout.get_virtual_bits().items() if self.input_qubit_mapping[k] < self._input_qubit_count})

    def initial_index_layout(self, filter_ancillas: bool=False) -> List[int]:
        """Generate an initial layout as an array of integers

        Args:
            filter_ancillas: If set to ``True`` any ancilla qubits added
                to the transpiler will not be included in the output.

        Return:
            A layout array that maps a position in the array to its new position in the output
            circuit.
        """
        virtual_map = self.initial_layout.get_virtual_bits()
        if filter_ancillas:
            output = [None] * self._input_qubit_count
        else:
            output = [None] * len(virtual_map)
        for index, (virt, phys) in enumerate(virtual_map.items()):
            if filter_ancillas and index >= self._input_qubit_count:
                break
            pos = self.input_qubit_mapping[virt]
            output[pos] = phys
        return output

    def routing_permutation(self) -> List[int]:
        """Generate a final layout as an array of integers

        If there is no :attr:`.final_layout` attribute present then that indicates
        there was no output permutation caused by routing or other transpiler
        transforms. In this case the function will return a list of ``[0, 1, 2, .., n]``
        to indicate this

        Returns:
            A layout array that maps a position in the array to its new position in the output
            circuit
        """
        if self.final_layout is None:
            return list(range(len(self._output_qubit_list)))
        virtual_map = self.final_layout.get_virtual_bits()
        return [virtual_map[virt] for virt in self._output_qubit_list]

    def final_index_layout(self, filter_ancillas: bool=True) -> List[int]:
        """Generate the final layout as an array of integers

        This method will generate an array of final positions for each qubit in the output circuit.
        For example, if you had an input circuit like::

            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)

        and the output from the transpiler was::

            tqc = QuantumCircuit(3)
            qc.h(2)
            qc.cx(2, 1)
            qc.swap(0, 1)
            qc.cx(2, 1)

        then the return from this function would be a list of::

            [2, 0, 1]

        because qubit 0 in the original circuit's final state is on qubit 3 in the output circuit,
        qubit 1 in the original circuit's final state is on qubit 0, and qubit 2's final state is
        on qubit. The output list length will be as wide as the input circuit's number of qubits,
        as the output list from this method is for tracking the permutation of qubits in the
        original circuit caused by the transpiler.

        Args:
            filter_ancillas: If set to ``False`` any ancillas allocated in the output circuit will be
                included in the layout.

        Returns:
            A list of final positions for each input circuit qubit
        """
        if self._input_qubit_count is None:
            num_source_qubits = len([x for x in self.input_qubit_mapping if getattr(x, '_register', '').startswith('ancilla')])
        else:
            num_source_qubits = self._input_qubit_count
        if self._output_qubit_list is None:
            circuit_qubits = list(self.final_layout.get_virtual_bits())
        else:
            circuit_qubits = self._output_qubit_list
        pos_to_virt = {v: k for k, v in self.input_qubit_mapping.items()}
        qubit_indices = []
        if filter_ancillas:
            num_qubits = num_source_qubits
        else:
            num_qubits = len(self._output_qubit_list)
        for index in range(num_qubits):
            qubit_idx = self.initial_layout[pos_to_virt[index]]
            if self.final_layout is not None:
                qubit_idx = self.final_layout[circuit_qubits[qubit_idx]]
            qubit_indices.append(qubit_idx)
        return qubit_indices

    def final_virtual_layout(self, filter_ancillas: bool=True) -> Layout:
        """Generate the final layout as a :class:`.Layout` object

        This method will generate an array of final positions for each qubit in the output circuit.
        For example, if you had an input circuit like::

            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)

        and the output from the transpiler was::

            tqc = QuantumCircuit(3)
            qc.h(2)
            qc.cx(2, 1)
            qc.swap(0, 1)
            qc.cx(2, 1)

        then the return from this function would be a layout object::

            Layout({
                qc.qubits[0]: 2,
                qc.qubits[1]: 0,
                qc.qubits[2]: 1,
            })

        because qubit 0 in the original circuit's final state is on qubit 3 in the output circuit,
        qubit 1 in the original circuit's final state is on qubit 0, and qubit 2's final state is
        on qubit. The output list length will be as wide as the input circuit's number of qubits,
        as the output list from this method is for tracking the permutation of qubits in the
        original circuit caused by the transpiler.

        Args:
            filter_ancillas: If set to ``False`` any ancillas allocated in the output circuit will be
                included in the layout.

        Returns:
            A layout object mapping to the final positions for each qubit
        """
        res = self.final_index_layout(filter_ancillas=filter_ancillas)
        pos_to_virt = {v: k for k, v in self.input_qubit_mapping.items()}
        return Layout({pos_to_virt[index]: phys for index, phys in enumerate(res)})