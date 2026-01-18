import copy
import pprint
from typing import Union, List
import numpy
from qiskit.qobj.common import QobjDictField
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.common import QobjExperimentHeader
class QobjMeasurementOption:
    """An individual measurement option."""

    def __init__(self, name, params=None):
        """Instantiate a new QobjMeasurementOption object.

        Args:
            name (str): The name of the measurement option
            params (list): The parameters of the measurement option.
        """
        self.name = name
        if params is not None:
            self.params = params

    def to_dict(self):
        """Return a dict format representation of the QobjMeasurementOption.

        Returns:
            dict: The dictionary form of the QasmMeasurementOption.
        """
        out_dict = {'name': self.name}
        if hasattr(self, 'params'):
            out_dict['params'] = self.params
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QobjMeasurementOption object from a dictionary.

        Args:
            data (dict): A dictionary for the experiment config

        Returns:
            QobjMeasurementOption: The object from the input dictionary.
        """
        name = data.pop('name')
        return cls(name, **data)

    def __eq__(self, other):
        if isinstance(other, QobjMeasurementOption):
            if self.to_dict() == other.to_dict():
                return True
        return False