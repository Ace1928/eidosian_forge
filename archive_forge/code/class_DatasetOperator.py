import json
import typing
from functools import lru_cache
from typing import Dict, FrozenSet, Generic, List, Type, TypeVar
import numpy as np
import pennylane as qml
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group, h5py
from pennylane.operation import Operator, Tensor
from ._wires import wires_to_json
class DatasetOperator(Generic[Op], DatasetAttribute[HDF5Group, Op, Op]):
    """``DatasetAttribute`` for ``pennylane.operation.Operator`` classes.

    Supports all operator types that meet the following conditions:
        - The ``__init__()`` method matches the signature of ``Operator.__init__``,
            or any additional arguments are optional and do not affect the value of
            the operator
        - The ``data`` and ``wires`` attributes will produce an identical copy of
            operator if passed into the classes' ``__init__()`` method. Generally,
            this means ``__init__()`` do not mutate the ``identifiers`` and ``wires``
            arguments.
        - Hyperparameters are not used or are automatically derived by ``__init__()``.

    Almost all operators meet these conditions. This type also supports serializing the
    ``Hamiltonian`` and ``Tensor`` operators.
    """
    type_id = 'operator'

    @classmethod
    @lru_cache(1)
    def consumes_types(cls) -> FrozenSet[Type[Operator]]:
        return frozenset((Tensor, qml.QubitCarry, qml.QubitSum, qml.Hamiltonian, qml.QubitUnitary, qml.DiagonalQubitUnitary, qml.Hadamard, qml.PauliX, qml.PauliY, qml.PauliZ, qml.X, qml.Y, qml.Z, qml.T, qml.S, qml.SX, qml.CNOT, qml.CH, qml.SWAP, qml.ECR, qml.SISWAP, qml.CSWAP, qml.CCZ, qml.Toffoli, qml.WireCut, qml.Hermitian, qml.Projector, qml.MultiRZ, qml.IsingXX, qml.IsingYY, qml.IsingZZ, qml.IsingXY, qml.PSWAP, qml.CPhaseShift00, qml.CPhaseShift01, qml.CPhaseShift10, qml.RX, qml.RY, qml.RZ, qml.PhaseShift, qml.Rot, qml.U1, qml.U2, qml.U3, qml.SingleExcitation, qml.SingleExcitationMinus, qml.SingleExcitationPlus, qml.DoubleExcitation, qml.DoubleExcitationMinus, qml.DoubleExcitationPlus, qml.OrbitalRotation, qml.FermionicSWAP, qml.SpecialUnitary, qml.BasisState, qml.QubitStateVector, qml.StatePrep, qml.QubitDensityMatrix, qml.QutritUnitary, qml.TShift, qml.TClock, qml.TAdd, qml.TSWAP, qml.THermitian, qml.AmplitudeDamping, qml.GeneralizedAmplitudeDamping, qml.PhaseDamping, qml.DepolarizingChannel, qml.BitFlip, qml.ResetError, qml.PauliError, qml.PhaseFlip, qml.ThermalRelaxationError, qml.Rotation, qml.Squeezing, qml.Displacement, qml.Beamsplitter, qml.TwoModeSqueezing, qml.QuadraticPhase, qml.ControlledAddition, qml.ControlledPhase, qml.Kerr, qml.CrossKerr, qml.InterferometerUnitary, qml.CoherentState, qml.SqueezedState, qml.DisplacedSqueezedState, qml.ThermalState, qml.GaussianState, qml.FockState, qml.FockStateVector, qml.FockDensityMatrix, qml.CatState, qml.NumberOperator, qml.TensorN, qml.QuadX, qml.QuadP, qml.QuadOperator, qml.PolyXP, qml.FockStateProjector, qml.Identity, qml.ControlledQubitUnitary, qml.ControlledPhaseShift, qml.CRX, qml.CRY, qml.CRZ, qml.CRot, qml.CZ, qml.CY))

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Op) -> HDF5Group:
        return self._ops_to_hdf5(bind_parent, key, [value])

    def hdf5_to_value(self, bind: HDF5Group) -> Op:
        return self._hdf5_to_ops(bind)[0]

    def _ops_to_hdf5(self, bind_parent: HDF5Group, key: str, value: typing.Sequence[Operator]) -> HDF5Group:
        """Serialize op sequence ``value``, and create nested sequences for any
        composite ops in ``value``.

        Since operators are commonly used in larger composite operations, we handle
        sequences of operators as the default case. This allows for performant (in
        time and space) serialization of large and nested operator sums, products, etc.
        """
        bind = bind_parent.create_group(key)
        op_wire_labels = []
        op_class_names = []
        for i, op in enumerate(value):
            op_key = f'op_{i}'
            if type(op) not in self.consumes_types():
                raise TypeError(f"Serialization of operator type '{type(op).__name__}' is not supported.")
            if isinstance(op, Tensor):
                self._ops_to_hdf5(bind, op_key, op.obs)
                op_wire_labels.append('null')
            elif isinstance(op, qml.Hamiltonian):
                coeffs, ops = op.terms()
                ham_grp = self._ops_to_hdf5(bind, op_key, ops)
                ham_grp['hamiltonian_coeffs'] = coeffs
                op_wire_labels.append('null')
            else:
                bind[op_key] = op.data if len(op.data) else h5py.Empty('f')
                op_wire_labels.append(wires_to_json(op.wires))
            op_class_names.append(type(op).__name__)
        bind['op_wire_labels'] = op_wire_labels
        bind['op_class_names'] = op_class_names
        return bind

    def _hdf5_to_ops(self, bind: HDF5Group) -> List[Operator]:
        """Load list of serialized ops from ``bind``."""
        ops = []
        names_bind = bind['op_class_names']
        wires_bind = bind['op_wire_labels']
        op_class_names = [] if names_bind.shape == (0,) else names_bind.asstr()
        op_wire_labels = [] if wires_bind.shape == (0,) else wires_bind.asstr()
        with qml.QueuingManager.stop_recording():
            for i, op_class_name in enumerate(op_class_names):
                op_key = f'op_{i}'
                op_cls = self._supported_ops_dict()[op_class_name]
                if op_cls is Tensor:
                    ops.append(Tensor(*self._hdf5_to_ops(bind[op_key])))
                elif op_cls is qml.Hamiltonian:
                    ops.append(qml.Hamiltonian(coeffs=list(bind[op_key]['hamiltonian_coeffs']), observables=self._hdf5_to_ops(bind[op_key])))
                else:
                    wire_labels = json.loads(op_wire_labels[i])
                    op_data = bind[op_key]
                    if op_data.shape is not None:
                        params = np.zeros(shape=op_data.shape, dtype=op_data.dtype)
                        op_data.read_direct(params)
                        ops.append(op_cls(*params, wires=wire_labels))
                    else:
                        ops.append(op_cls(wires=wire_labels))
        return ops

    @classmethod
    @lru_cache(1)
    def _supported_ops_dict(cls) -> Dict[str, Type[Operator]]:
        """Returns a dict mapping ``Operator`` subclass names to the class."""
        return {op.__name__: op for op in cls.consumes_types()}