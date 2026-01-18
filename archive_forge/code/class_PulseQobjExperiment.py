import copy
import pprint
from typing import Union, List
import numpy
from qiskit.qobj.common import QobjDictField
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.common import QobjExperimentHeader
class PulseQobjExperiment:
    """A Pulse Qobj Experiment.

    Each instance of this class is used to represent an individual Pulse
    experiment as part of a larger Pulse Qobj.
    """

    def __init__(self, instructions, config=None, header=None):
        """Instantiate a PulseQobjExperiment.

        Args:
            config (PulseQobjExperimentConfig): A config object for the experiment
            header (PulseQobjExperimentHeader): A header object for the experiment
            instructions (list): A list of :class:`PulseQobjInstruction` objects
        """
        if config is not None:
            self.config = config
        if header is not None:
            self.header = header
        self.instructions = instructions

    def to_dict(self):
        """Return a dictionary format representation of the Experiment.

        Returns:
            dict: The dictionary form of the PulseQobjExperiment.
        """
        out_dict = {'instructions': [x.to_dict() for x in self.instructions]}
        if hasattr(self, 'config'):
            out_dict['config'] = self.config.to_dict()
        if hasattr(self, 'header'):
            out_dict['header'] = self.header.to_dict()
        return out_dict

    def __repr__(self):
        instructions_str = [repr(x) for x in self.instructions]
        instructions_repr = '[' + ', '.join(instructions_str) + ']'
        out = 'PulseQobjExperiment('
        out += instructions_repr
        if hasattr(self, 'config') or hasattr(self, 'header'):
            out += ', '
        if hasattr(self, 'config'):
            out += 'config=' + str(repr(self.config)) + ', '
        if hasattr(self, 'header'):
            out += 'header=' + str(repr(self.header)) + ', '
        out += ')'
        return out

    def __str__(self):
        out = '\nPulse Experiment:\n'
        if hasattr(self, 'config'):
            config = pprint.pformat(self.config.to_dict())
        else:
            config = '{}'
        if hasattr(self, 'header'):
            header = pprint.pformat(self.header.to_dict() or {})
        else:
            header = '{}'
        out += 'Header:\n%s\n' % header
        out += 'Config:\n%s\n\n' % config
        for instruction in self.instructions:
            out += '\t%s\n' % instruction
        return out

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseQobjExperiment object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            PulseQobjExperiment: The object from the input dictionary.
        """
        config = None
        if 'config' in data:
            config = PulseQobjExperimentConfig.from_dict(data.pop('config'))
        header = None
        if 'header' in data:
            header = QobjExperimentHeader.from_dict(data.pop('header'))
        instructions = None
        if 'instructions' in data:
            instructions = [PulseQobjInstruction.from_dict(inst) for inst in data.pop('instructions')]
        return cls(instructions, config, header)

    def __eq__(self, other):
        if isinstance(other, PulseQobjExperiment):
            if self.to_dict() == other.to_dict():
                return True
        return False