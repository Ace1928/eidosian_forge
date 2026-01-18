import warnings
from typing import Sequence, Tuple, Optional
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from .measurements import AllCounts, Counts, SampleMeasurement
from .mid_measure import MeasurementValue
Groups the samples into a dictionary showing number of occurrences for
        each possible outcome.

        The format of the dictionary depends on the all_outcomes attribute. By default,
        the dictionary will only contain the observed outcomes. Optionally (all_outcomes=True)
        the dictionary will instead contain all possible outcomes, with a count of 0
        for those not observed. See example.

        Args:
            samples: An array of samples, with the shape being ``(shots,len(wires))`` if an observable
                is provided, with sample values being an array of 0s or 1s for each wire. Otherwise, it
                has shape ``(shots,)``, with sample values being scalar eigenvalues of the observable

        Returns:
            dict: dictionary with format ``{'outcome': num_occurrences}``, including all
                outcomes for the sampled observable

        **Example**

            >>> samples
            tensor([[0, 0],
                    [0, 0],
                    [1, 0]], requires_grad=True)

            By default, this will return:
            >>> self._samples_to_counts(samples)
            {'00': 2, '10': 1}

            However, if ``all_outcomes=True``, this will return:
            >>> self._samples_to_counts(samples)
            {'00': 2, '01': 0, '10': 1, '11': 0}

            The variable all_outcomes can be set when running measurements.counts, i.e.:

             .. code-block:: python3

                dev = qml.device("default.qubit", wires=2, shots=4)

                @qml.qnode(dev)
                def circuit(x):
                    qml.RX(x, wires=0)
                    return qml.counts(all_outcomes=True)

        