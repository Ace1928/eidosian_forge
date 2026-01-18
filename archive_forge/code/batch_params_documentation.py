from typing import Callable, Sequence
import pennylane as qml
from .core import transform
Transform a QNode to support an initial batch dimension
    for operation parameters.

    .. note::

        This transform will create multiple circuits inside the QNode, one per batch dimension.
        As a result, it is both simulator and hardware compatible. When using
        a simulator device, however, this means that a separate simulation
        will be performed per batch dimension.

    .. warning::

        Currently, not all templates have been updated to support a batch
        dimension. If you run into an error attempting to use a template
        with this transform, please open a GitHub issue detailing
        the error.

    Args:
        tape (QNode or QuantumTape or Callable): a quantum circuit to add a batch dimension to
        all_operations (bool): If ``True``, a batch dimension will be added to *all* operations
            in the QNode, rather than just trainable QNode parameters.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the batched results, with the first dimension treated as the batch dimension.

    **Example**

    Consider the following circuit:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.batch_params
        @qml.qnode(dev)
        def circuit(x, weights):
            qml.RX(x, wires=0)
            qml.RY(0.2, wires=1)
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.expval(qml.Hadamard(0))

    The ``qml.batch_params`` decorator allows us to pass arguments ``x`` and ``weights``
    that have a batch dimension. For example,

    >>> batch_size = 3
    >>> x = np.linspace(0.1, 0.5, batch_size)
    >>> rng = np.random.default_rng(seed=1234)
    >>> weights = rng.random((batch_size, 10, 3, 3), requires_grad=True)

    If we evaluate the QNode with these inputs, we will get an output
    of shape ``(batch_size,)``:

    >>> circuit(x, weights)
    tensor([ 0.00800498,  0.2735391 , -0.24395442], requires_grad=True)

    QNodes with a batch dimension remain fully differentiable:

    >>> cost_fn = lambda x, weights: np.sum(circuit(x, weights))
    >>> cost_fn(x, weights)
    tensor(0.03758966, requires_grad=True)
    >>> qml.grad(cost_fn)(x, weights)[0]
    array([-0.30262974,  0.06320878,  0.00811555])

    If we pass the ``all_operations`` argument, we can specify that
    *all* operation parameters in the transformed QNode, regardless of whether they
    are QNode input parameters, have a batch dimension:

    .. code-block:: python

        from functools import partial

        @partial(qml.batch_params, all_operations=True)
        @qml.qnode(dev)
        def circuit(x, weights):
            qml.RX(x, wires=0)
            qml.RY([0.2, 0.2, 0.2], wires=1)
            qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
            return qml.expval(qml.Hadamard(0))

    >>> cost_fn = lambda x, weights: np.sum(circuit(x, weights))
    >>> weights.requires_grad = False
    >>> cost_fn(x, weights)
    tensor(0.03758966, requires_grad=True)
    >>> qml.grad(cost_fn)(x, weights)[0]
    -0.30262974103192636
    