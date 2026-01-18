import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader
class QasmQobjExperiment:
    """An OpenQASM 2 Qobj Experiment.

    Each instance of this class is used to represent an OpenQASM 2 experiment as
    part of a larger OpenQASM 2 qobj.
    """

    def __init__(self, config=None, header=None, instructions=None):
        """Instantiate a QasmQobjExperiment.

        Args:
            config (QasmQobjExperimentConfig): A config object for the experiment
            header (QasmQobjExperimentHeader): A header object for the experiment
            instructions (list): A list of :class:`QasmQobjInstruction` objects
        """
        self.config = config or QasmQobjExperimentConfig()
        self.header = header or QasmQobjExperimentHeader()
        self.instructions = instructions or []

    def __repr__(self):
        instructions_str = [repr(x) for x in self.instructions]
        instructions_repr = '[' + ', '.join(instructions_str) + ']'
        out = 'QasmQobjExperiment(config={}, header={}, instructions={})'.format(repr(self.config), repr(self.header), instructions_repr)
        return out

    def __str__(self):
        out = '\nOpenQASM2 Experiment:\n'
        config = pprint.pformat(self.config.to_dict())
        header = pprint.pformat(self.header.to_dict())
        out += 'Header:\n%s\n' % header
        out += 'Config:\n%s\n\n' % config
        for instruction in self.instructions:
            out += '\t%s\n' % instruction
        return out

    def to_dict(self):
        """Return a dictionary format representation of the Experiment.

        Returns:
            dict: The dictionary form of the QasmQObjExperiment.
        """
        out_dict = {'config': self.config.to_dict(), 'header': self.header.to_dict(), 'instructions': [x.to_dict() for x in self.instructions]}
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QasmQobjExperiment object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            QasmQobjExperiment: The object from the input dictionary.
        """
        config = None
        if 'config' in data:
            config = QasmQobjExperimentConfig.from_dict(data.pop('config'))
        header = None
        if 'header' in data:
            header = QasmQobjExperimentHeader.from_dict(data.pop('header'))
        instructions = None
        if 'instructions' in data:
            instructions = [QasmQobjInstruction.from_dict(inst) for inst in data.pop('instructions')]
        return cls(config, header, instructions)

    def __eq__(self, other):
        if isinstance(other, QasmQobjExperiment):
            if self.to_dict() == other.to_dict():
                return True
        return False