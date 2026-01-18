import copy
import pprint
from typing import Union, List
import numpy
from qiskit.qobj.common import QobjDictField
from qiskit.qobj.common import QobjHeader
from qiskit.qobj.common import QobjExperimentHeader
class PulseQobjExperimentConfig(QobjDictField):
    """A config for a single Pulse experiment in the qobj."""

    def __init__(self, qubit_lo_freq=None, meas_lo_freq=None, **kwargs):
        """Instantiate a PulseQobjExperimentConfig object.

        Args:
            qubit_lo_freq (List[float]): List of qubit LO frequencies in GHz.
            meas_lo_freq (List[float]): List of meas readout LO frequencies in GHz.
            kwargs: Additional free form key value fields to add to the configuration
        """
        if qubit_lo_freq is not None:
            self.qubit_lo_freq = qubit_lo_freq
        if meas_lo_freq is not None:
            self.meas_lo_freq = meas_lo_freq
        if kwargs:
            self.__dict__.update(kwargs)