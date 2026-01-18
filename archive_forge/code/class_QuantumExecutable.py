import abc
import dataclasses
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Sequence, cast, Dict, Any, List, Iterator
import cirq
from cirq import _compat, study
@dataclass(frozen=True)
class QuantumExecutable:
    """An executable quantum program.

    This serves a similar purpose to `cirq.Circuit` with some key differences. First, a quantum
    executable contains all the relevant context for execution including parameters as well as
    the desired number of repetitions. Second, this object is immutable. Finally, there are
    optional fields enabling a higher level of abstraction for certain aspects of the executable.

    Attributes:
        circuit: A `cirq.Circuit` describing the quantum operations to execute.
        measurement: A description of the measurement properties or process.
        params: An immutable `cirq.ParamResolver` (or similar type). It's representation is
            normalized to a tuple of key value pairs.
        spec: Optional `cg.ExecutableSpec` containing metadata about this executable that is not
            used by the quantum runtime, but will be forwarded to all downstream result objects.
        problem_topology: Optional `cirq.NamedTopology` instance specifying the topology of the
            circuit. This is useful when optimizing on-device layout. If none is provided we
            assume `circuit` already has a valid on-device layout.
        initial_state: A `cirq.ProductState` specifying the desired initial state before executing
            `circuit`. If not specified, default to the all-zeros state.
    """
    circuit: cirq.FrozenCircuit
    measurement: BitstringsMeasurement
    params: Optional[Tuple[TParamPair, ...]] = None
    spec: Optional[ExecutableSpec] = None
    problem_topology: Optional[cirq.NamedTopology] = None
    initial_state: Optional[cirq.ProductState] = None

    def __init__(self, circuit: cirq.AbstractCircuit, measurement: BitstringsMeasurement, params: Union[Sequence[TParamPair], cirq.ParamResolverOrSimilarType]=None, spec: Optional[ExecutableSpec]=None, problem_topology: Optional[cirq.NamedTopology]=None, initial_state: Optional[cirq.ProductState]=None):
        """Initialize the quantum executable.

        The actual fields in this class are immutable, but we allow more liberal input types
        which will be frozen in this __init__ method.

        Args:
            circuit: The circuit. This will be frozen before being set as an attribute.
            measurement: A description of the measurement properties or process.
            params: A cirq.ParamResolverOrSimilarType which will be frozen into a tuple of
                key value pairs.
            spec: Specification metadata about this executable that is not used by the quantum
                runtime, but is persisted in result objects to associate executables with results.
            problem_topology: Description of the multiqubit gate topology present in the circuit.
                If not specified, the circuit must be compatible with the device topology.
            initial_state: How to initialize the quantum system before running `circuit`. If not
                specified, the device will be initialized into the all-zeros state.
        """
        object.__setattr__(self, 'circuit', circuit.freeze())
        object.__setattr__(self, 'measurement', measurement)
        if isinstance(params, tuple) and all((isinstance(param_kv, tuple) and len(param_kv) == 2 for param_kv in params)):
            frozen_params = params
        elif isinstance(params, Sequence) and all((isinstance(param_kv, Sequence) and len(param_kv) == 2 for param_kv in params)):
            frozen_params = tuple(((k, v) for k, v in params))
        elif study.resolver._is_param_resolver_or_similar_type(params):
            param_resolver = cirq.ParamResolver(cast(cirq.ParamResolverOrSimilarType, params))
            frozen_params = tuple(param_resolver.param_dict.items())
        else:
            raise ValueError(f'`params` should be a ParamResolverOrSimilarType, not {params}.')
        object.__setattr__(self, 'params', frozen_params)
        object.__setattr__(self, 'spec', spec)
        object.__setattr__(self, 'problem_topology', problem_topology)
        object.__setattr__(self, 'initial_state', initial_state)
        object.__setattr__(self, '_hash', hash(dataclasses.astuple(self)))

    def __str__(self):
        return f'QuantumExecutable(spec={self.spec})'

    def __repr__(self):
        return _compat.dataclass_repr(self, namespace='cirq_google')

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)