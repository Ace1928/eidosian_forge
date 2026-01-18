from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
class gradient_transform(qml.batch_transform):
    """Decorator for defining quantum gradient transforms.

    Quantum gradient transforms are a specific case of :class:`~.batch_transform`.
    All quantum gradient transforms accept a tape, and output
    a batch of tapes to be independently executed on a quantum device, alongside
    a post-processing function that returns the result.

    Args:
        expand_fn (function): An expansion function (if required) to be applied to the
            input tape before the gradient computation takes place. If not provided,
            the default expansion function simply expands all operations that
            have ``Operation.grad_method=None`` until all resulting operations
            have a defined gradient method.
        differentiable (bool): Specifies whether the gradient transform is differentiable or
            not. A transform may be non-differentiable if it does not use an
            autodiff framework for its tensor manipulations. In such a case, setting
            ``differentiable=False`` instructs the decorator
            to mark the output as 'constant', reducing potential overhead.
        hybrid (bool): Specifies whether classical processing inside a QNode
            should be taken into account when transforming a QNode.

            - If ``True``, and classical processing is detected and this
              option is set to ``True``, the Jacobian of the classical
              processing will be computed and included. When evaluated, the
              returned Jacobian will be with respect to the QNode arguments.

            - If ``False``, any internal QNode classical processing will be
              **ignored**. When evaluated, the returned Jacobian will be with
              respect to the **gate** arguments, and not the QNode arguments.

    Supported gradient transforms must be of the following form:

    .. code-block:: python

        @gradient_transform
        def my_custom_gradient(tape, argnum=None, **kwargs):
            ...
            return gradient_tapes, processing_fn

    where:

    - ``tape`` (*QuantumTape*): the input quantum tape to compute the gradient of

    - ``argnum`` (*int* or *list[int]* or *None*): Which trainable parameters of the tape
      to differentiate with respect to. If not provided, the derivatives with respect to all
      trainable inputs of the tape should be returned (``tape.trainable_params``).

    - ``gradient_tapes`` (*list[QuantumTape]*): is a list of output tapes to be evaluated.
      If this list is empty, no quantum evaluations will be made.

    - ``processing_fn`` is a processing function to be applied to the output of the evaluated
      ``gradient_tapes``. It should accept a list of numeric results with length ``len(gradient_tapes)``,
      and return the Jacobian matrix.

    Once defined, the quantum gradient transform can be used as follows:

    >>> gradient_tapes, processing_fn = my_custom_gradient(tape, *gradient_kwargs)
    >>> res = execute(tapes, dev, interface="autograd", gradient_fn=qml.gradients.param_shift)
    >>> jacobian = processing_fn(res)

    Alternatively, gradient transforms can be applied directly to QNodes,
    in which case the execution is implicit:

    >>> fn = my_custom_gradient(qnode, *gradient_kwargs)
    >>> fn(weights) # transformed function takes the same arguments as the QNode
    1.2629730888100839

    .. note::

        The input tape might have parameters of various types, including
        NumPy arrays, JAX Arrays, and TensorFlow and PyTorch tensors.

        If the gradient transform is written in a autodiff-compatible manner, either by
        using a framework such as Autograd or TensorFlow, or by using ``qml.math`` for
        tensor manipulation, then higher-order derivatives will also be supported.

        Alternatively, you may use the ``tape.unwrap()`` context manager to temporarily
        convert all tape parameters to NumPy arrays and floats:

        >>> with tape.unwrap():
        ...     params = tape.get_parameters()  # list of floats
    """

    def __repr__(self):
        return f'<gradient_transform: {self.__name__}>'

    def __init__(self, transform_fn, expand_fn=expand_invalid_trainable, differentiable=True, hybrid=True):
        self.hybrid = hybrid
        super().__init__(transform_fn, expand_fn=expand_fn, differentiable=differentiable)

    def default_qnode_wrapper(self, qnode, targs, tkwargs):
        hybrid = tkwargs.pop('hybrid', self.hybrid)
        _wrapper = super().default_qnode_wrapper(qnode, targs, tkwargs)

        def jacobian_wrapper(*args, **kwargs):
            argnums = tkwargs.get('argnums', None)
            interface = qml.math.get_interface(*args)
            trainable_params = qml.math.get_trainable_indices(args)
            if interface == 'jax' and tkwargs.get('argnum', None):
                raise qml.QuantumFunctionError('argnum does not work with the Jax interface. You should use argnums instead.')
            if interface == 'jax' and (not trainable_params):
                if argnums is None:
                    argnums_ = [0]
                else:
                    argnums_ = [argnums] if isinstance(argnums, int) else argnums
                params = qml.math.jax_argnums_to_tape_trainable(qnode, argnums_, self.expand_fn, args, kwargs)
                argnums_ = qml.math.get_trainable_indices(params)
                kwargs['argnums'] = argnums_
            elif not trainable_params:
                warnings.warn('Attempted to compute the gradient of a QNode with no trainable parameters. If this is unintended, please add trainable parameters in accordance with the chosen auto differentiation framework.')
                return ()
            qjac = _wrapper(*args, **kwargs)
            if not hybrid:
                return qjac
            kwargs.pop('shots', False)
            argnum_cjac = trainable_params or argnums if interface == 'jax' else None
            cjac = qml.gradients.classical_jacobian(qnode, argnum=argnum_cjac, expand_fn=self.expand_fn)(*args, **kwargs)
            return _contract_qjac_with_cjac(qjac, cjac, qnode.tape)
        return jacobian_wrapper