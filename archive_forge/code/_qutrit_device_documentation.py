import itertools
import numpy as np
import pennylane as qml
from pennylane import QubitDevice
from pennylane.measurements import MeasurementProcess
from pennylane.wires import Wires
Group the obtained samples into a dictionary.

            **Example**

                >>> samples
                tensor([[0, 0, 1],
                        [0, 0, 1],
                        [1, 1, 1]], requires_grad=True)
                >>> self._samples_to_counts(samples)
                {'111':1, '001':2}
            