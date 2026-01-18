import abc
import inspect
import logging
from typing import Tuple, Callable, Optional, Union
from cachetools import LRUCache
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike
class LightningVJPs(DeviceDerivatives):
    """Calculates VJPs natively using lightning.qubit.

    Args:
        device (LightningBase): a device in the lightning ecosystem (``lightning.qubit``, ``lightning.gpu``, ``lightning.kokkos``.)
        gradient_kwargs (Optional[dict]):  Any gradient options.

    >>> dev = qml.device('lightning.qubit', wires=5)
    >>> jpc = LightningVJPs(dev, gradient_kwargs={"use_device_state": True, "method": "adjoint_jacobian"})
    >>> tape = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.Z(0))])
    >>> dev.batch_execute((tape,))
    [array(0.36235775)]
    >>> jpc.compute_vjp((tape,), (0.5,) )
    ((array(-0.46601954),),)
    >>> -0.5 * np.sin(1.2)
    -0.46601954298361314

    """

    def __repr__(self):
        return f'<LightningVJPs: {self._device.short_name}, {self._gradient_kwargs}>'

    def __init__(self, device, gradient_kwargs=None):
        super().__init__(device, gradient_kwargs=gradient_kwargs)
        self._processed_gradient_kwargs = {key: value for key, value in self._gradient_kwargs.items() if key != 'method'}

    def compute_vjp(self, tapes, dy):
        if not all((isinstance(m, qml.measurements.ExpectationMP) for t in tapes for m in t.measurements)):
            raise NotImplementedError('Lightning device VJPs only support expectation values.')
        results = []
        for dyi, tape in zip(dy, tapes):
            numpy_tape = qml.transforms.convert_to_numpy_parameters(tape)
            if len(tape.measurements) == 1:
                dyi = (dyi,)
            dyi = np.array(qml.math.unwrap(dyi))
            if qml.math.ndim(dyi) > 1:
                raise NotImplementedError('Lightning device VJPs are not supported with jax jacobians.')
            vjp_f = self._device.vjp(numpy_tape.measurements, dyi, **self._processed_gradient_kwargs)
            out = vjp_f(numpy_tape)
            if len(tape.trainable_params) == 1:
                out = (out,)
            results.append(out)
        return tuple(results)