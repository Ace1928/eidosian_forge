import warnings
from typing import Sequence, Tuple, Optional
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from .measurements import AllCounts, Counts, SampleMeasurement
from .mid_measure import MeasurementValue
def _samples_to_counts(self, samples):
    """Groups the samples into a dictionary showing number of occurrences for
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

        """
    outcomes = []
    batched_ndims = 2
    shape = qml.math.shape(samples)
    if self.obs is None and (not isinstance(self.mv, MeasurementValue)):
        mask = qml.math.isnan(samples)
        num_wires = shape[-1]
        if np.any(mask):
            mask = np.logical_not(np.any(mask, axis=tuple(range(1, samples.ndim))))
            samples = samples[mask, ...]

        def convert(x):
            return f'{x:0{num_wires}b}'
        exp2 = 2 ** np.arange(num_wires - 1, -1, -1)
        samples = np.einsum('...i,i', samples, exp2)
        new_shape = samples.shape
        samples = qml.math.cast_like(samples, qml.math.int8(0))
        samples = list(map(convert, samples.ravel()))
        samples = np.array(samples).reshape(new_shape)
        batched_ndims = 3
        if self.all_outcomes:
            num_wires = len(self.wires) if len(self.wires) > 0 else shape[-1]
            outcomes = list(map(convert, range(2 ** num_wires)))
    elif self.all_outcomes:
        outcomes = self.eigvals()
    batched = len(shape) == batched_ndims
    if not batched:
        samples = samples[None]
    base_dict = {k: qml.math.int64(0) for k in outcomes}
    outcome_dicts = [base_dict.copy() for _ in range(shape[0])]
    results = [qml.math.unique(batch, return_counts=True) for batch in samples]
    for result, outcome_dict in zip(results, outcome_dicts):
        states, _counts = result
        for state, count in zip(qml.math.unwrap(states), _counts):
            outcome_dict[state] = count
    return outcome_dicts if batched else outcome_dicts[0]