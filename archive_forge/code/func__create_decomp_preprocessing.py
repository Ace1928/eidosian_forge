import contextlib
import pennylane as qml
from pennylane.operation import (
def _create_decomp_preprocessing(custom_decomps, dev, decomp_depth=10):
    """Creates a custom preprocessing method for a device that applies
    a set of specified custom decompositions.

    Args:
        custom_decomps (Dict[Union(str, qml.operation.Operation), Callable]): Custom
            decompositions to be applied by the device at runtime.
        dev (pennylane.devices.Device): A quantum device.
        decomp_depth: The maximum depth of the expansion.

    Returns:
        Callable: A custom preprocessing method that a device can call to expand
        its tapes.

    **Example**

    Suppose we would like a custom expansion function that decomposes all CNOTs
    into CZs. We first define a decomposition function:

    .. code-block:: python

        def custom_cnot(wires):
            return [
                qml.Hadamard(wires=wires[1]),
                qml.CZ(wires=[wires[0], wires[1]]),
                qml.Hadamard(wires=wires[1])
            ]

    We then create the custom function (passing a device, in order to pick up any
    additional stopping criteria the expansion should have), and then register the
    result as a custom function of the device:

    >>> custom_decomps = {qml.CNOT : custom_cnot}
    >>> new_preprocessing = _create_decomp_preprocessing(custom_decomps, dev)
    >>> dev.preprocess = new_preprocessing
    """

    def decomposer(op):
        if isinstance(op, qml.ops.Controlled) and type(op.base) in custom_decomps:
            op.base.compute_decomposition = custom_decomps[type(op.base)]
            return op.decomposition()
        if op.name in custom_decomps:
            return custom_decomps[op.name](*op.data, wires=op.wires, **op.hyperparameters)
        if type(op) in custom_decomps:
            return custom_decomps[type(op)](*op.data, wires=op.wires, **op.hyperparameters)
        return op.decomposition()
    original_preprocess = dev.preprocess

    def new_preprocess(execution_config=qml.devices.DefaultExecutionConfig):
        program, config = original_preprocess(execution_config)
        for container in program:
            if container.transform == qml.devices.preprocess.decompose.transform:
                container.kwargs['decomposer'] = decomposer
                container.kwargs['max_expansion'] = decomp_depth
                for cond in ['stopping_condition', 'stopping_condition_shots']:
                    original_stopping_condition = container.kwargs[cond]

                    def stopping_condition(obj):
                        if obj.name in custom_decomps or type(obj) in custom_decomps:
                            return False
                        return original_stopping_condition(obj)
                    container.kwargs[cond] = stopping_condition
                break
        return (program, config)
    return new_preprocess