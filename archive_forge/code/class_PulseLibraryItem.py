import copy
import pprint
from typing import Union, List
import numpy
from qiskit.qobj.common import QobjDictField
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.common import QobjExperimentHeader
class PulseLibraryItem:
    """An item in a pulse library."""

    def __init__(self, name, samples):
        """Instantiate a pulse library item.

        Args:
            name (str): A name for the pulse.
            samples (list[complex]): A list of complex values defining pulse
                shape.
        """
        self.name = name
        if isinstance(samples[0], list):
            self.samples = numpy.array([complex(sample[0], sample[1]) for sample in samples])
        else:
            self.samples = samples

    def to_dict(self):
        """Return a dictionary format representation of the pulse library item.

        Returns:
            dict: The dictionary form of the PulseLibraryItem.
        """
        return {'name': self.name, 'samples': self.samples}

    @classmethod
    def from_dict(cls, data):
        """Create a new PulseLibraryItem object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            PulseLibraryItem: The object from the input dictionary.
        """
        return cls(**data)

    def __repr__(self):
        return f'PulseLibraryItem({self.name}, {repr(self.samples)})'

    def __str__(self):
        return f'Pulse Library Item:\n\tname: {self.name}\n\tsamples: {self.samples}'

    def __eq__(self, other):
        if isinstance(other, PulseLibraryItem):
            if self.to_dict() == other.to_dict():
                return True
        return False