from functools import wraps
import inspect
from typing import Union, Callable, Tuple
import pennylane as qml
from .qnode import QNode, _make_execution_config, _get_device_shots
def _get_full_transform_program(qnode: QNode) -> 'qml.transforms.core.TransformProgram':
    program = qml.transforms.core.TransformProgram(qnode.transform_program)
    if getattr(qnode.gradient_fn, 'expand_transform', False):
        program.add_transform(qml.transform(qnode.gradient_fn.expand_transform), **qnode.gradient_kwargs)
    if isinstance(qnode.device, qml.devices.Device):
        config = _make_execution_config(qnode)
        return program + qnode.device.preprocess(config)[0]
    program.add_transform(qml.transform(qnode.device.batch_transform))
    program.add_transform(expand_fn_transform(qnode.device.expand_fn))
    return program