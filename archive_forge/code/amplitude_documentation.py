import pennylane as qml
from pennylane.ops import StatePrep
from pennylane.wires import Wires
Validate and pre-process inputs as follows:

        * If features is batched, the processing that follows is applied to each feature set in the batch.
        * Check that the features tensor is one-dimensional.
        * If pad_with is None, check that the last dimension of the features tensor
          has length :math:`2^n` where :math:`n` is the number of qubits. Else check that the
          last dimension of the features tensor is not larger than :math:`2^n` and pad features
          with value if necessary.
        * If normalize is false, check that last dimension of features is normalised to one. Else, normalise the
          features tensor.
        