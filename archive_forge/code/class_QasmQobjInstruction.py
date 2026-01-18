import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader
class QasmQobjInstruction:
    """A class representing a single instruction in an QasmQobj Experiment."""

    def __init__(self, name, params=None, qubits=None, register=None, memory=None, condition=None, conditional=None, label=None, mask=None, relation=None, val=None, snapshot_type=None):
        """Instantiate a new QasmQobjInstruction object.

        Args:
            name (str): The name of the instruction
            params (list): The list of parameters for the gate
            qubits (list): A list of ``int`` representing the qubits the
                instruction operates on
            register (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of register slots in which to
                store the measurement results (must be the same length as
                qubits). If a ``bfunc`` instruction this is a single ``int``
                of the register slot in which to store the result.
            memory (list): If a ``measure`` instruction this is a list
                of ``int`` containing the list of memory slots to store the
                measurement results in (must be the same length as qubits).
                If a ``bfunc`` instruction this is a single ``int`` of the
                memory slot to store the boolean function result in.
            condition (tuple): A tuple of the form ``(int, int)`` where the
                first ``int`` is the control register and the second ``int`` is
                the control value if the gate has a condition.
            conditional (int):  The register index of the condition
            label (str): An optional label assigned to the instruction
            mask (int): For a ``bfunc`` instruction the hex value which is
                applied as an ``AND`` to the register bits.
            relation (str): Relational  operator  for  comparing  the  masked
                register to the ``val`` kwarg. Can be either ``==`` (equals) or
                ``!=`` (not equals).
            val (int): Value to which to compare the masked register. In other
                words, the output of the function is ``(register AND mask)``
            snapshot_type (str): For snapshot instructions the type of snapshot
                to use
        """
        self.name = name
        if params is not None:
            self.params = params
        if qubits is not None:
            self.qubits = qubits
        if register is not None:
            self.register = register
        if memory is not None:
            self.memory = memory
        if condition is not None:
            self._condition = condition
        if conditional is not None:
            self.conditional = conditional
        if label is not None:
            self.label = label
        if mask is not None:
            self.mask = mask
        if relation is not None:
            self.relation = relation
        if val is not None:
            self.val = val
        if snapshot_type is not None:
            self.snapshot_type = snapshot_type

    def to_dict(self):
        """Return a dictionary format representation of the Instruction.

        Returns:
            dict: The dictionary form of the QasmQobjInstruction.
        """
        out_dict = {'name': self.name}
        for attr in ['params', 'qubits', 'register', 'memory', '_condition', 'conditional', 'label', 'mask', 'relation', 'val', 'snapshot_type']:
            if hasattr(self, attr):
                if attr == 'params':
                    params = []
                    for param in list(getattr(self, attr)):
                        if isinstance(param, ParameterExpression):
                            params.append(float(param))
                        else:
                            params.append(param)
                    out_dict[attr] = params
                else:
                    out_dict[attr] = getattr(self, attr)
        return out_dict

    def __repr__(self):
        out = "QasmQobjInstruction(name='%s'" % self.name
        for attr in ['params', 'qubits', 'register', 'memory', '_condition', 'conditional', 'label', 'mask', 'relation', 'val', 'snapshot_type']:
            attr_val = getattr(self, attr, None)
            if attr_val is not None:
                if isinstance(attr_val, str):
                    out += f', {attr}="{attr_val}"'
                else:
                    out += f', {attr}={attr_val}'
        out += ')'
        return out

    def __str__(self):
        out = 'Instruction: %s\n' % self.name
        for attr in ['params', 'qubits', 'register', 'memory', '_condition', 'conditional', 'label', 'mask', 'relation', 'val', 'snapshot_type']:
            if hasattr(self, attr):
                out += f'\t\t{attr}: {getattr(self, attr)}\n'
        return out

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjInstruction object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            QasmQobjInstruction: The object from the input dictionary.
        """
        name = data.pop('name')
        return cls(name, **data)

    def __eq__(self, other):
        if isinstance(other, QasmQobjInstruction):
            if self.to_dict() == other.to_dict():
                return True
        return False