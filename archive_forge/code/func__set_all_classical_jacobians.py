from functools import partial
from typing import Callable, List, Tuple, Optional, Sequence, Union
import pennylane as qml
from pennylane.typing import Result, ResultBatch
from pennylane.tape import QuantumTape
from .transform_dispatcher import TransformContainer, TransformError, TransformDispatcher
def _set_all_classical_jacobians(self, qnode, args, kwargs, argnums):
    """It can be called inside the QNode to get all the classical Jacobians for a gradient transform."""

    def classical_preprocessing(program, *args, **kwargs):
        """Returns the trainable gate parameters for a given QNode input."""
        kwargs.pop('shots', None)
        kwargs.pop('argnums', None)
        qnode.construct(args, kwargs)
        tape = qnode.qtape
        tapes, _ = program((tape,))
        res = tuple((qml.math.stack(tape.get_parameters(trainable_only=True)) for tape in tapes))
        if len(tapes) == 1:
            return res[0]
        return res

    def jacobian(classical_function, program, argnums, *args, **kwargs):
        indices = qml.math.get_trainable_indices(args)
        if qnode.interface in ['jax', 'jax-jit']:
            import jax
            if isinstance(args[0], jax.numpy.ndarray):
                argnums = 0 if argnums is None else argnums
        if not indices and argnums is None:
            raise qml.QuantumFunctionError('No trainable parameters.')
        classical_function = partial(classical_function, program)
        if qnode.interface == 'autograd':
            jac = qml.jacobian(classical_function, argnum=argnums)(*args, **kwargs)
        if qnode.interface == 'tf':
            import tensorflow as tf

            def _jacobian(*args, **kwargs):
                with tf.GradientTape() as tape:
                    gate_params = classical_function(*args, **kwargs)
                jac = tape.jacobian(gate_params, args)
                return jac
            jac = _jacobian(*args, **kwargs)
        if qnode.interface == 'torch':
            import torch

            def _jacobian(*args, **kwargs):
                jac = torch.autograd.functional.jacobian(classical_function, args)
                return jac
            jac = _jacobian(*args, **kwargs)
        if qnode.interface in ['jax', 'jax-jit']:
            import jax
            argnums = 0 if argnums is None else argnums

            def _jacobian(*args, **kwargs):
                return jax.jacobian(classical_function, argnums=argnums)(*args, **kwargs)
            jac = _jacobian(*args, **kwargs)
        return jac
    classical_jacobians = []
    for index, transform in enumerate(self):
        if transform.classical_cotransform:
            argnum = transform._kwargs.get('argnum', None)
            if qnode.interface == 'jax' and argnum:
                raise qml.QuantumFunctionError('argnum does not work with the Jax interface. You should use argnums instead.')
            sub_program = TransformProgram(self[0:index])
            classical_jacobian = jacobian(classical_preprocessing, sub_program, argnums, *args, **kwargs)
            qnode.construct(args, kwargs)
            tapes, _ = sub_program((qnode.tape,))
            multi_tapes = len(tapes) > 1
            if not multi_tapes:
                classical_jacobian = [classical_jacobian]
            classical_jacobians.append(classical_jacobian)
        else:
            classical_jacobians.append(None)
    self._classical_jacobians = classical_jacobians
    qnode.construct(args, kwargs)