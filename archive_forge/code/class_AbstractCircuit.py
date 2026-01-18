import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
class AbstractCircuit(abc.ABC):
    """The base class for Circuit-like objects.

    A circuit-like object must have a list of moments (which can be empty).

    These methods return information about the circuit, and can be called on
    either Circuit or FrozenCircuit objects:

    *   next_moment_operating_on
    *   prev_moment_operating_on
    *   next_moments_operating_on
    *   operation_at
    *   all_qubits
    *   all_operations
    *   findall_operations
    *   findall_operations_between
    *   findall_operations_until_blocked
    *   findall_operations_with_gate_type
    *   reachable_frontier_from
    *   has_measurements
    *   are_all_matches_terminal
    *   are_all_measurements_terminal
    *   unitary
    *   final_state_vector
    *   to_text_diagram
    *   to_text_diagram_drawer
    *   qid_shape
    *   all_measurement_key_names
    *   to_quil
    *   to_qasm
    *   save_qasm
    *   get_independent_qubit_sets
    """

    @classmethod
    def from_moments(cls: Type[CIRCUIT_TYPE], *moments: 'cirq.OP_TREE') -> CIRCUIT_TYPE:
        """Create a circuit from moment op trees.

        Args:
            *moments: Op tree for each moment. If an op tree is a moment, it
                will be included directly in the new circuit. If an op tree is
                a circuit, it will be frozen, wrapped in a CircuitOperation, and
                included in its own moment in the new circuit. Otherwise, the
                op tree will be passed to `cirq.Moment` to create a new moment
                which is then included in the new circuit. Note that in the
                latter case we have the normal restriction that operations in a
                moment must be applied to disjoint sets of qubits.
        """
        return cls._from_moments(cls._make_moments(moments))

    @staticmethod
    def _make_moments(moments: Iterable['cirq.OP_TREE']) -> Iterator['cirq.Moment']:
        for m in moments:
            if isinstance(m, Moment):
                yield m
            elif isinstance(m, AbstractCircuit):
                yield Moment(m.freeze().to_op())
            else:
                yield Moment(m)

    @classmethod
    @abc.abstractmethod
    def _from_moments(cls: Type[CIRCUIT_TYPE], moments: Iterable['cirq.Moment']) -> CIRCUIT_TYPE:
        """Create a circuit from moments.

        This must be implemented by subclasses. It provides a more efficient way
        to construct a circuit instance since we already have the moments and so
        can skip the analysis required to implement various insert strategies.

        Args:
            moments: Moments of the circuit.
        """

    @property
    @abc.abstractmethod
    def moments(self) -> Sequence['cirq.Moment']:
        pass

    @abc.abstractmethod
    def freeze(self) -> 'cirq.FrozenCircuit':
        """Creates a FrozenCircuit from this circuit.

        If 'self' is a FrozenCircuit, the original object is returned.
        """

    @abc.abstractmethod
    def unfreeze(self, copy: bool=True) -> 'cirq.Circuit':
        """Creates a Circuit from this circuit.

        Args:
            copy: If True and 'self' is a Circuit, returns a copy that circuit.
        """

    def __bool__(self):
        return bool(self.moments)

    def __eq__(self, other):
        if not isinstance(other, AbstractCircuit):
            return NotImplemented
        return tuple(self.moments) == tuple(other.moments)

    def _approx_eq_(self, other: Any, atol: Union[int, float]) -> bool:
        """See `cirq.protocols.SupportsApproximateEquality`."""
        if not isinstance(other, AbstractCircuit):
            return NotImplemented
        return cirq.protocols.approx_eq(tuple(self.moments), tuple(other.moments), atol=atol)

    def __ne__(self, other) -> bool:
        return not self == other

    def __len__(self) -> int:
        return len(self.moments)

    def __iter__(self) -> Iterator['cirq.Moment']:
        return iter(self.moments)

    def _decompose_(self) -> 'cirq.OP_TREE':
        """See `cirq.SupportsDecompose`."""
        return self.all_operations()

    @overload
    def __getitem__(self, key: int) -> 'cirq.Moment':
        pass

    @overload
    def __getitem__(self, key: Tuple[int, 'cirq.Qid']) -> 'cirq.Operation':
        pass

    @overload
    def __getitem__(self, key: Tuple[int, Iterable['cirq.Qid']]) -> 'cirq.Moment':
        pass

    @overload
    def __getitem__(self, key: slice) -> Self:
        pass

    @overload
    def __getitem__(self, key: Tuple[slice, 'cirq.Qid']) -> Self:
        pass

    @overload
    def __getitem__(self, key: Tuple[slice, Iterable['cirq.Qid']]) -> Self:
        pass

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._from_moments(self.moments[key])
        if hasattr(key, '__index__'):
            return self.moments[key]
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('If key is tuple, it must be a pair.')
            moment_idx, qubit_idx = key
            selected_moments = self.moments[moment_idx]
            if isinstance(selected_moments, Moment):
                return selected_moments[qubit_idx]
            if isinstance(qubit_idx, ops.Qid):
                qubit_idx = [qubit_idx]
            return self._from_moments((moment[qubit_idx] for moment in selected_moments))
        raise TypeError('__getitem__ called with key not of type slice, int, or tuple.')

    def __str__(self) -> str:
        return self.to_text_diagram()

    def _repr_args(self) -> str:
        args = []
        if self.moments:
            args.append(_list_repr_with_indented_item_lines(self.moments))
        return f'{', '.join(args)}'

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f'cirq.{cls_name}({self._repr_args()})'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Print ASCII diagram in Jupyter."""
        cls_name = self.__class__.__name__
        if cycle:
            p.text(f'{cls_name}(...)')
        else:
            p.text(self.to_text_diagram())

    def _repr_html_(self) -> str:
        """Print ASCII diagram in Jupyter notebook without wrapping lines."""
        return '<pre style="overflow: auto; white-space: pre;">' + html.escape(self.to_text_diagram()) + '</pre>'

    def _first_moment_operating_on(self, qubits: Iterable['cirq.Qid'], indices: Iterable[int]) -> Optional[int]:
        qubits = frozenset(qubits)
        for m in indices:
            if self._has_op_at(m, qubits):
                return m
        return None

    def next_moment_operating_on(self, qubits: Iterable['cirq.Qid'], start_moment_index: int=0, max_distance: Optional[int]=None) -> Optional[int]:
        """Finds the index of the next moment that touches the given qubits.

        Args:
            qubits: We're looking for operations affecting any of these qubits.
            start_moment_index: The starting point of the search.
            max_distance: The number of moments (starting from the start index
                and moving forward) to check. Defaults to no limit.

        Returns:
            None if there is no matching moment, otherwise the index of the
            earliest matching moment.

        Raises:
          ValueError: negative max_distance.
        """
        max_circuit_distance = len(self.moments) - start_moment_index
        if max_distance is None:
            max_distance = max_circuit_distance
        elif max_distance < 0:
            raise ValueError(f'Negative max_distance: {max_distance}')
        else:
            max_distance = min(max_distance, max_circuit_distance)
        return self._first_moment_operating_on(qubits, range(start_moment_index, start_moment_index + max_distance))

    def next_moments_operating_on(self, qubits: Iterable['cirq.Qid'], start_moment_index: int=0) -> Dict['cirq.Qid', int]:
        """Finds the index of the next moment that touches each qubit.

        Args:
            qubits: The qubits to find the next moments acting on.
            start_moment_index: The starting point of the search.

        Returns:
            The index of the next moment that touches each qubit. If there
            is no such moment, the next moment is specified as the number of
            moments in the circuit. Equivalently, can be characterized as one
            plus the index of the last moment after start_moment_index
            (inclusive) that does *not* act on a given qubit.
        """
        next_moments = {}
        for q in qubits:
            next_moment = self.next_moment_operating_on([q], start_moment_index)
            next_moments[q] = len(self.moments) if next_moment is None else next_moment
        return next_moments

    def prev_moment_operating_on(self, qubits: Sequence['cirq.Qid'], end_moment_index: Optional[int]=None, max_distance: Optional[int]=None) -> Optional[int]:
        """Finds the index of the previous moment that touches the given qubits.

        Args:
            qubits: We're looking for operations affecting any of these qubits.
            end_moment_index: The moment index just after the starting point of
                the reverse search. Defaults to the length of the list of
                moments.
            max_distance: The number of moments (starting just before from the
                end index and moving backward) to check. Defaults to no limit.

        Returns:
            None if there is no matching moment, otherwise the index of the
            latest matching moment.

        Raises:
            ValueError: negative max_distance.
        """
        if end_moment_index is None:
            end_moment_index = len(self.moments)
        if max_distance is None:
            max_distance = len(self.moments)
        elif max_distance < 0:
            raise ValueError(f'Negative max_distance: {max_distance}')
        else:
            max_distance = min(end_moment_index, max_distance)
        if end_moment_index > len(self.moments):
            d = end_moment_index - len(self.moments)
            end_moment_index -= d
            max_distance -= d
        if max_distance <= 0:
            return None
        return self._first_moment_operating_on(qubits, (end_moment_index - k - 1 for k in range(max_distance)))

    def reachable_frontier_from(self, start_frontier: Dict['cirq.Qid', int], *, is_blocker: Callable[['cirq.Operation'], bool]=lambda op: False) -> Dict['cirq.Qid', int]:
        """Determines how far can be reached into a circuit under certain rules.

        The location L = (qubit, moment_index) is *reachable* if and only if the
        following all hold true:

        - There is not a blocking operation covering L.
        -  At least one of the following holds:
            - qubit is in start frontier and moment_index =
                max(start_frontier[qubit], 0).
            - There is no operation at L and prev(L) = (qubit,
                moment_index-1) is reachable.
            - There is an (non-blocking) operation P covering L such that
                (q', moment_index - 1) is reachable for every q' on which P
                acts.

        An operation in moment moment_index is blocking if at least one of the
        following hold:

        - `is_blocker` returns a truthy value.
        - The operation acts on a qubit not in start_frontier.
        - The operation acts on a qubit q such that start_frontier[q] >
            moment_index.

        In other words, the reachable region extends forward through time along
        each qubit in start_frontier until it hits a blocking operation. Any
        location involving a qubit not in start_frontier is unreachable.

        For each qubit q in `start_frontier`, the reachable locations will
        correspond to a contiguous range starting at start_frontier[q] and
        ending just before some index end_q. The result of this method is a
        dictionary, and that dictionary maps each qubit q to its end_q.

        Examples:

        If `start_frontier` is

        ```
        {
            cirq.LineQubit(0): 6,
            cirq.LineQubit(1): 2,
            cirq.LineQubit(2): 2
        }
        ```

        then the reachable wire locations in the following circuit are
        highlighted with '█' characters:

        ```

                0   1   2   3   4   5   6   7   8   9   10  11  12  13
            0: ───H───@─────────────────█████████████████████─@───H───
                      │                                       │
            1: ───────@─██H███@██████████████████████─@───H───@───────
                              │                       │
            2: ─────────██████@███H██─@───────@───H───@───────────────
                                      │       │
            3: ───────────────────────@───H───@───────────────────────
        ```

        And the computed `end_frontier` is

        ```
        {
            cirq.LineQubit(0): 11,
            cirq.LineQubit(1): 9,
            cirq.LineQubit(2): 6,
        }
        ```

        Note that the frontier indices (shown above the circuit) are
        best thought of (and shown) as happening *between* moment indices.

        If we specify a blocker as follows:

        ```
        is_blocker=lambda: op == cirq.CZ(cirq.LineQubit(1),
                                         cirq.LineQubit(2))
        ```

        and use this `start_frontier`:

        ```
        {
            cirq.LineQubit(0): 0,
            cirq.LineQubit(1): 0,
            cirq.LineQubit(2): 0,
            cirq.LineQubit(3): 0,
        }
        ```

        Then this is the reachable area:

        ```

                0   1   2   3   4   5   6   7   8   9   10  11  12  13
            0: ─██H███@██████████████████████████████████████─@───H───
                      │                                       │
            1: ─██████@███H██─@───────────────────────@───H───@───────
                              │                       │
            2: ─█████████████─@───H───@───────@───H───@───────────────
                                      │       │
            3: ─█████████████████████─@───H───@───────────────────────

        ```

        and the computed `end_frontier` is:

        ```
        {
            cirq.LineQubit(0): 11,
            cirq.LineQubit(1): 3,
            cirq.LineQubit(2): 3,
            cirq.LineQubit(3): 5,
        }
        ```

        Args:
            start_frontier: A starting set of reachable locations.
            is_blocker: A predicate that determines if operations block
                reachability. Any location covered by an operation that causes
                `is_blocker` to return True is considered to be an unreachable
                location.

        Returns:
            An end_frontier dictionary, containing an end index for each qubit q
            mapped to a start index by the given `start_frontier` dictionary.

            To determine if a location (q, i) was reachable, you can use
            this expression:

                q in start_frontier and start_frontier[q] <= i < end_frontier[q]

            where i is the moment index, q is the qubit, and end_frontier is the
            result of this method.
        """
        active: Set['cirq.Qid'] = set()
        end_frontier = {}
        queue = BucketPriorityQueue[ops.Operation](drop_duplicate_entries=True)

        def enqueue_next(qubit: 'cirq.Qid', moment: int) -> None:
            next_moment = self.next_moment_operating_on([qubit], moment)
            if next_moment is None:
                end_frontier[qubit] = max(len(self), start_frontier[qubit])
                if qubit in active:
                    active.remove(qubit)
            else:
                next_op = self.operation_at(qubit, next_moment)
                assert next_op is not None
                queue.enqueue(next_moment, next_op)
        for start_qubit, start_moment in start_frontier.items():
            enqueue_next(start_qubit, start_moment)
        while queue:
            cur_moment, cur_op = queue.dequeue()
            for q in cur_op.qubits:
                if q in start_frontier and cur_moment >= start_frontier[q] and (q not in end_frontier):
                    active.add(q)
            continue_past = cur_op is not None and active.issuperset(cur_op.qubits) and (not is_blocker(cur_op))
            if continue_past:
                for q in cur_op.qubits:
                    enqueue_next(q, cur_moment + 1)
            else:
                for q in cur_op.qubits:
                    if q in active:
                        end_frontier[q] = cur_moment
                        active.remove(q)
        return end_frontier

    def findall_operations_between(self, start_frontier: Dict['cirq.Qid', int], end_frontier: Dict['cirq.Qid', int], omit_crossing_operations: bool=False) -> List[Tuple[int, 'cirq.Operation']]:
        """Finds operations between the two given frontiers.

        If a qubit is in `start_frontier` but not `end_frontier`, its end index
        defaults to the end of the circuit. If a qubit is in `end_frontier` but
        not `start_frontier`, its start index defaults to the start of the
        circuit. Operations on qubits not mentioned in either frontier are not
        included in the results.

        Args:
            start_frontier: Just before where to start searching for operations,
                for each qubit of interest. Start frontier indices are
                inclusive.
            end_frontier: Just before where to stop searching for operations,
                for each qubit of interest. End frontier indices are exclusive.
            omit_crossing_operations: Determines whether or not operations that
                cross from a location between the two frontiers to a location
                outside the two frontiers are included or excluded. (Operations
                completely inside are always included, and operations completely
                outside are always excluded.)

        Returns:
            A list of tuples. Each tuple describes an operation found between
            the two frontiers. The first item of each tuple is the index of the
            moment containing the operation, and the second item is the
            operation itself. The list is sorted so that the moment index
            increases monotonically.
        """
        result = BucketPriorityQueue[ops.Operation](drop_duplicate_entries=True)
        involved_qubits = set(start_frontier.keys()) | set(end_frontier.keys())
        for q in sorted(involved_qubits):
            for i in range(start_frontier.get(q, 0), end_frontier.get(q, len(self))):
                op = self.operation_at(q, i)
                if op is None:
                    continue
                if omit_crossing_operations and (not involved_qubits.issuperset(op.qubits)):
                    continue
                result.enqueue(i, op)
        return list(result)

    def findall_operations_until_blocked(self, start_frontier: Dict['cirq.Qid', int], is_blocker: Callable[['cirq.Operation'], bool]=lambda op: False) -> List[Tuple[int, 'cirq.Operation']]:
        """Finds all operations until a blocking operation is hit.

        An operation is considered blocking if both of the following hold:

        - It is in the 'light cone' of start_frontier.
        - `is_blocker` returns a truthy value, or it acts on a blocked qubit

        Every qubit acted on by a blocking operation is thereafter itself
        blocked.

        The notion of reachability here differs from that in
        reachable_frontier_from in two respects:

        - An operation is not considered blocking only because it is in a
            moment before the start_frontier of one of the qubits on which it
            acts.
        - Operations that act on qubits not in start_frontier are not
            automatically blocking.

        For every (moment_index, operation) returned:

        - moment_index >= min((start_frontier[q] for q in operation.qubits
            if q in start_frontier), default=0)
        - set(operation.qubits).intersection(start_frontier)

        Below are some examples, where on the left the opening parentheses show
        `start_frontier` and on the right are the operations included (with
        their moment indices) in the output. `F` and `T` indicate that
        `is_blocker` return `False` or `True`, respectively, when applied to
        the gates; `M` indicates that it doesn't matter.

        ```
            ─(─F───F───────    ┄(─F───F─)┄┄┄┄┄
               │   │              │   │
            ─(─F───F───T─── => ┄(─F───F─)┄┄┄┄┄
                       │                  ┊
            ───────────T───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


            ───M─────(─F───    ┄┄┄┄┄┄┄┄┄(─F─)┄┄
               │       │          ┊       │
            ───M───M─(─F───    ┄┄┄┄┄┄┄┄┄(─F─)┄┄
                   │        =>        ┊
            ───────M───M───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
                       │                  ┊
            ───────────M───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


            ───M─(─────M───     ┄┄┄┄┄()┄┄┄┄┄┄┄┄
               │       │           ┊       ┊
            ───M─(─T───M───     ┄┄┄┄┄()┄┄┄┄┄┄┄┄
                   │        =>         ┊
            ───────T───M───     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
                       │                   ┊
            ───────────M───     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


            ─(─F───F───    ┄(─F───F─)┄
               │   │    =>    │   │
            ───F─(─F───    ┄(─F───F─)┄


            ─(─F───────────    ┄(─F─)┄┄┄┄┄┄┄┄┄
               │                  │
            ───F───F───────    ┄(─F─)┄┄┄┄┄┄┄┄┄
                   │        =>        ┊
            ───────F───F───    ┄┄┄┄┄┄┄┄┄(─F─)┄
                       │                  │
            ─(─────────F───    ┄┄┄┄┄┄┄┄┄(─F─)┄
        ```

        Args:
            start_frontier: A starting set of reachable locations.
            is_blocker: A predicate that determines if operations block
                reachability. Any location covered by an operation that causes
                `is_blocker` to return True is considered to be an unreachable
                location.

        Returns:
            A list of tuples. Each tuple describes an operation found between
            the start frontier and a blocking operation. The first item of
            each tuple is the index of the moment containing the operation,
            and the second item is the operation itself.

        """
        op_list: List[Tuple[int, ops.Operation]] = []
        if not start_frontier:
            return op_list
        start_index = min(start_frontier.values())
        blocked_qubits: Set[cirq.Qid] = set()
        for index, moment in enumerate(self[start_index:], start_index):
            active_qubits = set((q for q, s in start_frontier.items() if s <= index))
            for op in moment.operations:
                if is_blocker(op) or blocked_qubits.intersection(op.qubits):
                    blocked_qubits.update(op.qubits)
                elif active_qubits.intersection(op.qubits):
                    op_list.append((index, op))
            if blocked_qubits.issuperset(start_frontier):
                break
        return op_list

    def operation_at(self, qubit: 'cirq.Qid', moment_index: int) -> Optional['cirq.Operation']:
        """Finds the operation on a qubit within a moment, if any.

        Args:
            qubit: The qubit to check for an operation on.
            moment_index: The index of the moment to check for an operation
                within. Allowed to be beyond the end of the circuit.

        Returns:
            None if there is no operation on the qubit at the given moment, or
            else the operation.
        """
        if not 0 <= moment_index < len(self.moments):
            return None
        return self.moments[moment_index].operation_at(qubit)

    def findall_operations(self, predicate: Callable[['cirq.Operation'], bool]) -> Iterable[Tuple[int, 'cirq.Operation']]:
        """Find the locations of all operations that satisfy a given condition.

        This returns an iterator of (index, operation) tuples where each
        operation satisfies op_cond(operation) is truthy. The indices are
        in order of the moments and then order of the ops within that moment.

        Args:
            predicate: A method that takes an Operation and returns a Truthy
                value indicating the operation meets the find condition.

        Returns:
            An iterator (index, operation)'s that satisfy the op_condition.
        """
        for index, moment in enumerate(self.moments):
            for op in moment.operations:
                if predicate(op):
                    yield (index, op)

    def findall_operations_with_gate_type(self, gate_type: Type[_TGate]) -> Iterable[Tuple[int, 'cirq.GateOperation', _TGate]]:
        """Find the locations of all gate operations of a given type.

        Args:
            gate_type: The type of gate to find, e.g. XPowGate or
                MeasurementGate.

        Returns:
            An iterator (index, operation, gate)'s for operations with the given
            gate type.
        """
        result = self.findall_operations(lambda operation: isinstance(operation.gate, gate_type))
        for index, op in result:
            gate_op = cast(ops.GateOperation, op)
            yield (index, gate_op, cast(_TGate, gate_op.gate))

    def has_measurements(self):
        """Returns whether or not this circuit has measurements.

        Returns: True if `cirq.is_measurement(self)` is True otherwise False.
        """
        return protocols.is_measurement(self)

    def _is_measurement_(self) -> bool:
        return any((protocols.is_measurement(op) for op in self.all_operations()))

    def are_all_measurements_terminal(self) -> bool:
        """Whether all measurement gates are at the end of the circuit.

        Returns: True iff no measurement is followed by a gate.
        """
        return self.are_all_matches_terminal(protocols.is_measurement)

    def are_all_matches_terminal(self, predicate: Callable[['cirq.Operation'], bool]) -> bool:
        """Check whether all of the ops that satisfy a predicate are terminal.

        This method will transparently descend into any CircuitOperations this
        circuit contains; as a result, it will misbehave if the predicate
        refers to CircuitOperations. See the tests for an example of this.

        Args:
            predicate: A predicate on ops.Operations which is being checked.

        Returns:
            Whether or not all `Operation` s in a circuit that satisfy the
            given predicate are terminal. Also checks within any CircuitGates
            the circuit may contain.
        """
        from cirq.circuits import CircuitOperation
        if not all((self.next_moment_operating_on(op.qubits, i + 1) is None for i, op in self.findall_operations(predicate) if not isinstance(op.untagged, CircuitOperation))):
            return False
        for i, moment in enumerate(self.moments):
            for op in moment.operations:
                circuit = getattr(op.untagged, 'circuit', None)
                if circuit is None:
                    continue
                if not circuit.are_all_matches_terminal(predicate):
                    return False
                if i < len(self.moments) - 1 and (not all((self.next_moment_operating_on(op.qubits, i + 1) is None for _, op in circuit.findall_operations(predicate)))):
                    return False
        return True

    def are_any_measurements_terminal(self) -> bool:
        """Whether any measurement gates are at the end of the circuit.

        Returns: True iff some measurements are not followed by a gate.
        """
        return self.are_any_matches_terminal(protocols.is_measurement)

    def are_any_matches_terminal(self, predicate: Callable[['cirq.Operation'], bool]) -> bool:
        """Check whether any of the ops that satisfy a predicate are terminal.

        This method will transparently descend into any CircuitOperations this
        circuit contains; as a result, it will misbehave if the predicate
        refers to CircuitOperations. See the tests for an example of this.

        Args:
            predicate: A predicate on ops.Operations which is being checked.

        Returns:
            Whether or not any `Operation` s in a circuit that satisfy the
            given predicate are terminal. Also checks within any CircuitGates
            the circuit may contain.
        """
        from cirq.circuits import CircuitOperation
        if any((self.next_moment_operating_on(op.qubits, i + 1) is None for i, op in self.findall_operations(predicate) if not isinstance(op.untagged, CircuitOperation))):
            return True
        for i, moment in reversed(list(enumerate(self.moments))):
            for op in moment.operations:
                circuit = getattr(op.untagged, 'circuit', None)
                if circuit is None:
                    continue
                if not circuit.are_any_matches_terminal(predicate):
                    continue
                if i == len(self.moments) - 1 or any((self.next_moment_operating_on(op.qubits, i + 1) is None for _, op in circuit.findall_operations(predicate))):
                    return True
        return False

    def _has_op_at(self, moment_index: int, qubits: Iterable['cirq.Qid']) -> bool:
        return 0 <= moment_index < len(self.moments) and self.moments[moment_index].operates_on(qubits)

    def all_qubits(self) -> FrozenSet['cirq.Qid']:
        """Returns the qubits acted upon by Operations in this circuit.

        Returns: FrozenSet of `cirq.Qid` objects acted on by all operations
            in this circuit.
        """
        return frozenset((q for m in self.moments for q in m.qubits))

    def all_operations(self) -> Iterator['cirq.Operation']:
        """Returns an iterator over the operations in the circuit.

        Returns: Iterator over `cirq.Operation` elements found in this circuit.
        """
        return (op for moment in self for op in moment.operations)

    def map_operations(self, func: Callable[['cirq.Operation'], 'cirq.OP_TREE']) -> Self:
        """Applies the given function to all operations in this circuit.

        Args:
            func: a mapping function from operations to OP_TREEs.

        Returns:
            A circuit with the same basic structure as the original, but with
            each operation `op` replaced with `func(op)`.
        """

        def map_moment(moment: 'cirq.Moment') -> 'cirq.Circuit':
            """Apply func to expand each op into a circuit, then zip up the circuits."""
            return Circuit.zip(*[Circuit(func(op)) for op in moment])
        return self._from_moments((m for moment in self for m in map_moment(moment)))

    def qid_shape(self, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> Tuple[int, ...]:
        """Get the qubit shapes of all qubits in this circuit.

        Returns: A tuple containing the dimensions (shape) of all qudits
            found in this circuit according to `qubit_order`.
        """
        qids = ops.QubitOrder.as_qubit_order(qubit_order).order_for(self.all_qubits())
        return protocols.qid_shape(qids)

    def all_measurement_key_objs(self) -> FrozenSet['cirq.MeasurementKey']:
        return frozenset((key for op in self.all_operations() for key in protocols.measurement_key_objs(op)))

    def _measurement_key_objs_(self) -> FrozenSet['cirq.MeasurementKey']:
        """Returns the set of all measurement keys in this circuit.

        Returns: FrozenSet of `cirq.MeasurementKey` objects that are
            in this circuit.
        """
        return self.all_measurement_key_objs()

    def all_measurement_key_names(self) -> FrozenSet[str]:
        """Returns the set of all measurement key names in this circuit.

        Returns: FrozenSet of strings that are the measurement key
            names in this circuit.
        """
        return frozenset((key for op in self.all_operations() for key in protocols.measurement_key_names(op)))

    def _measurement_key_names_(self) -> FrozenSet[str]:
        return self.all_measurement_key_names()

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]):
        return self._from_moments((protocols.with_measurement_key_mapping(moment, key_map) for moment in self.moments))

    def _with_key_path_(self, path: Tuple[str, ...]):
        return self._from_moments((protocols.with_key_path(moment, path) for moment in self.moments))

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
        return self._from_moments((protocols.with_key_path_prefix(moment, prefix) for moment in self.moments))

    def _with_rescoped_keys_(self, path: Tuple[str, ...], bindable_keys: FrozenSet['cirq.MeasurementKey']):
        moments = []
        for moment in self.moments:
            new_moment = protocols.with_rescoped_keys(moment, path, bindable_keys)
            moments.append(new_moment)
            bindable_keys |= protocols.measurement_key_objs(new_moment)
        return self._from_moments(moments)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self.qid_shape()

    def _has_unitary_(self) -> bool:
        if not self.are_all_measurements_terminal():
            return False
        unitary_ops = protocols.decompose(self.all_operations(), keep=protocols.has_unitary, intercepting_decomposer=_decompose_measurement_inversions, on_stuck_raise=None)
        return all((protocols.has_unitary(e) for e in unitary_ops))

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        """Converts the circuit into a unitary matrix, if possible.

        If the circuit contains any non-terminal measurements, the conversion
        into a unitary matrix fails (i.e. returns NotImplemented). Terminal
        measurements are ignored when computing the unitary matrix. The unitary
        matrix is the product of the unitary matrix of all operations in the
        circuit (after expanding them to apply to the whole system).
        """
        if not self._has_unitary_():
            return NotImplemented
        return self.unitary(ignore_terminal_measurements=True)

    def unitary(self, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, qubits_that_should_be_present: Iterable['cirq.Qid']=(), ignore_terminal_measurements: bool=True, dtype: Type[np.complexfloating]=np.complex128) -> np.ndarray:
        """Converts the circuit into a unitary matrix, if possible.

        Returns the same result as `cirq.unitary`, but provides more options.

        Args:
            qubit_order: Determines how qubits are ordered when passing matrices
                into np.kron.
            qubits_that_should_be_present: Qubits that may or may not appear
                in operations within the circuit, but that should be included
                regardless when generating the matrix.
            ignore_terminal_measurements: When set, measurements at the end of
                the circuit are ignored instead of causing the method to
                fail.
            dtype: The numpy dtype for the returned unitary. Defaults to
                np.complex128. Specifying np.complex64 will run faster at the
                cost of precision. `dtype` must be a complex np.dtype, unless
                all operations in the circuit have unitary matrices with
                exclusively real coefficients (e.g. an H + TOFFOLI circuit).

        Returns:
            A (possibly gigantic) 2d numpy array corresponding to a matrix
            equivalent to the circuit's effect on a quantum state.

        Raises:
            ValueError: The circuit contains measurement gates that are not
                ignored.
            TypeError: The circuit contains gates that don't have a known
                unitary matrix, e.g. gates parameterized by a Symbol.
        """
        if not ignore_terminal_measurements and any((protocols.is_measurement(op) for op in self.all_operations())):
            raise ValueError('Circuit contains a measurement.')
        if not self.are_all_measurements_terminal():
            raise ValueError('Circuit contains a non-terminal measurement.')
        qs = ops.QubitOrder.as_qubit_order(qubit_order).order_for(self.all_qubits().union(qubits_that_should_be_present))
        qid_shape = self.qid_shape(qubit_order=qs)
        side_len = np.prod(qid_shape, dtype=np.int64)
        state = qis.eye_tensor(qid_shape, dtype=dtype)
        result = _apply_unitary_circuit(self, state, qs, dtype)
        return result.reshape((side_len, side_len))

    def _has_superoperator_(self) -> bool:
        """Returns True if self has superoperator representation."""
        return all((m._has_superoperator_() for m in self))

    def _superoperator_(self) -> np.ndarray:
        """Compute superoperator matrix for quantum channel specified by this circuit."""
        all_qubits = self.all_qubits()
        n = len(all_qubits)
        if n > 10:
            raise ValueError(f'{n} > 10 qubits is too many to compute superoperator')
        circuit_superoperator = np.eye(4 ** n)
        for moment in self:
            full_moment = moment.expand_to(all_qubits)
            moment_superoperator = full_moment._superoperator_()
            circuit_superoperator = moment_superoperator @ circuit_superoperator
        return circuit_superoperator

    def final_state_vector(self, *, initial_state: 'cirq.STATE_VECTOR_LIKE'=0, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, ignore_terminal_measurements: bool=False, dtype: Type[np.complexfloating]=np.complex128, param_resolver: 'cirq.ParamResolverOrSimilarType'=None, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> np.ndarray:
        """Returns the state vector resulting from acting operations on a state.

        This is equivalent to calling cirq.final_state_vector with the same
        arguments and this circuit as the "program".

        Args:
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state. Otherwise  if this
                is a np.ndarray it is the full initial state. In this case it
                must be the correct size, be normalized (an L2 norm of 1), and
                be safely castable to an appropriate dtype for the simulator.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            qubits_that_should_be_present: Qubits that may or may not appear
                in operations within the circuit, but that should be included
                regardless when generating the matrix.
            ignore_terminal_measurements: When set, measurements at the end of
                the circuit are ignored instead of causing the method to
                fail. Defaults to False.
            dtype: The `numpy.dtype` used by the simulation. Typically one of
                `numpy.complex64` or `numpy.complex128`.
            param_resolver: Parameters to run with the program.
            seed: The random seed to use for this simulator.

        Returns:
            The state vector resulting from applying the given unitary
            operations to the desired initial state. Specifically, a numpy
            array containing the amplitudes in np.kron order, where the
            order of arguments to kron is determined by the qubit order
            argument (which defaults to just sorting the qubits that are
            present into an ascending order).

        Raises:
            ValueError: If the program doesn't have a well defined final state
                because it has non-unitary gates.
        """
        from cirq.sim.mux import final_state_vector
        return final_state_vector(self, initial_state=initial_state, param_resolver=param_resolver, qubit_order=qubit_order, ignore_terminal_measurements=ignore_terminal_measurements, dtype=dtype, seed=seed)

    def to_text_diagram(self, *, use_unicode_characters: bool=True, transpose: bool=False, include_tags: bool=True, precision: Optional[int]=3, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> str:
        """Returns text containing a diagram describing the circuit.

        Args:
            use_unicode_characters: Determines if unicode characters are
                allowed (as opposed to ascii-only diagrams).
            transpose: Arranges qubit wires vertically instead of horizontally.
            include_tags: Whether tags on TaggedOperations should be printed
            precision: Number of digits to display in text diagram
            qubit_order: Determines how qubits are ordered in the diagram.

        Returns:
            The text diagram.
        """
        diagram = self.to_text_diagram_drawer(use_unicode_characters=use_unicode_characters, include_tags=include_tags, precision=precision, qubit_order=qubit_order, transpose=transpose)
        return diagram.render(crossing_char=None if use_unicode_characters else '-' if transpose else '|', horizontal_spacing=1 if transpose else 3, use_unicode_characters=use_unicode_characters)

    def to_text_diagram_drawer(self, *, use_unicode_characters: bool=True, qubit_namer: Optional[Callable[['cirq.Qid'], str]]=None, transpose: bool=False, include_tags: bool=True, draw_moment_groups: bool=True, precision: Optional[int]=3, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT, get_circuit_diagram_info: Optional[Callable[['cirq.Operation', 'cirq.CircuitDiagramInfoArgs'], 'cirq.CircuitDiagramInfo']]=None) -> 'cirq.TextDiagramDrawer':
        """Returns a TextDiagramDrawer with the circuit drawn into it.

        Args:
            use_unicode_characters: Determines if unicode characters are
                allowed (as opposed to ascii-only diagrams).
            qubit_namer: Names qubits in diagram. Defaults to using _circuit_diagram_info_ or str.
            transpose: Arranges qubit wires vertically instead of horizontally.
            include_tags: Whether to include tags in the operation.
            draw_moment_groups: Whether to draw moment symbol or not
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the diagram.
            get_circuit_diagram_info: Gets circuit diagram info. Defaults to
                protocol with fallback.

        Returns:
            The TextDiagramDrawer instance.
        """
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(self.all_qubits())
        cbits = tuple(sorted(set((key for op in self.all_operations() for key in protocols.control_keys(op))), key=str))
        labels = qubits + cbits
        label_map = {labels[i]: i for i in range(len(labels))}

        def default_namer(label_entity):
            info = protocols.circuit_diagram_info(label_entity, default=None)
            qubit_name = info.wire_symbols[0] if info else str(label_entity)
            return qubit_name + ('' if transpose else ': ')
        if qubit_namer is None:
            qubit_namer = default_namer
        diagram = TextDiagramDrawer()
        diagram.write(0, 0, '')
        for label_entity, i in label_map.items():
            name = qubit_namer(label_entity) if isinstance(label_entity, ops.Qid) else default_namer(label_entity)
            diagram.write(0, i, name)
        first_annotation_row = max(label_map.values(), default=0) + 1
        if any((isinstance(op.gate, cirq.GlobalPhaseGate) for op in self.all_operations())):
            diagram.write(0, max(label_map.values(), default=0) + 1, 'global phase:')
            first_annotation_row += 1
        moment_groups: List[Tuple[int, int]] = []
        for moment in self.moments:
            _draw_moment_in_diagram(moment=moment, use_unicode_characters=use_unicode_characters, label_map=label_map, out_diagram=diagram, precision=precision, moment_groups=moment_groups, get_circuit_diagram_info=get_circuit_diagram_info, include_tags=include_tags, first_annotation_row=first_annotation_row, transpose=transpose)
        w = diagram.width()
        for i in label_map.values():
            diagram.horizontal_line(i, 0, w, doubled=not isinstance(labels[i], ops.Qid))
        if moment_groups and draw_moment_groups:
            _draw_moment_groups_in_diagram(moment_groups, use_unicode_characters, diagram)
        if transpose:
            diagram = diagram.transpose()
        return diagram

    def _is_parameterized_(self) -> bool:
        return any((protocols.is_parameterized(op) for op in self.all_operations()))

    def _parameter_names_(self) -> AbstractSet[str]:
        return {name for op in self.all_operations() for name in protocols.parameter_names(op)}

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> Self:
        changed = False
        resolved_moments: List['cirq.Moment'] = []
        for moment in self:
            resolved_moment = protocols.resolve_parameters(moment, resolver, recursive)
            if resolved_moment is not moment:
                changed = True
            resolved_moments.append(resolved_moment)
        if not changed:
            return self
        return self._from_moments(resolved_moments)

    def _qasm_(self) -> str:
        return self.to_qasm()

    def _to_qasm_output(self, header: Optional[str]=None, precision: int=10, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> 'cirq.QasmOutput':
        """Returns a QASM object equivalent to the circuit.

        Args:
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
        """
        if header is None:
            header = f'Generated from Cirq v{cirq._version.__version__}'
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(self.all_qubits())
        return QasmOutput(operations=self.all_operations(), qubits=qubits, header=header, precision=precision, version='2.0')

    def to_qasm(self, header: Optional[str]=None, precision: int=10, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> str:
        """Returns QASM equivalent to the circuit.

        Args:
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
        """
        return str(self._to_qasm_output(header, precision, qubit_order))

    def save_qasm(self, file_path: Union[str, bytes, int], header: Optional[str]=None, precision: int=10, qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> None:
        """Save a QASM file equivalent to the circuit.

        Args:
            file_path: The location of the file where the qasm will be written.
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
        """
        self._to_qasm_output(header, precision, qubit_order).save(file_path)

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['moments'])

    @classmethod
    def _from_json_dict_(cls, moments, **kwargs):
        return cls(moments, strategy=InsertStrategy.EARLIEST)

    def zip(*circuits: 'cirq.AbstractCircuit', align: Union['cirq.Alignment', str]=Alignment.LEFT) -> 'cirq.AbstractCircuit':
        """Combines operations from circuits in a moment-by-moment fashion.

        Moment k of the resulting circuit will have all operations from moment
        k of each of the given circuits.

        When the given circuits have different lengths, the shorter circuits are
        implicitly padded with empty moments. This differs from the behavior of
        python's built-in zip function, which would instead truncate the longer
        circuits.

        The zipped circuits can't have overlapping operations occurring at the
        same moment index.

        Args:
            *circuits: The circuits to merge together.
            align: The alignment for the zip, see `cirq.Alignment`.

        Returns:
            The merged circuit.

        Raises:
            ValueError: If the zipped circuits have overlapping operations occurring
                at the same moment index.

        Examples:

        >>> import cirq
        >>> a, b, c, d = cirq.LineQubit.range(4)
        >>> circuit1 = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b))
        >>> circuit2 = cirq.Circuit(cirq.X(c), cirq.Y(c), cirq.Z(c))
        >>> circuit3 = cirq.Circuit(cirq.Moment(), cirq.Moment(cirq.S(d)))
        >>> print(circuit1.zip(circuit2))
        0: ───H───@───────
                  │
        1: ───────X───────
        <BLANKLINE>
        2: ───X───Y───Z───
        >>> print(circuit1.zip(circuit2, circuit3))
        0: ───H───@───────
                  │
        1: ───────X───────
        <BLANKLINE>
        2: ───X───Y───Z───
        <BLANKLINE>
        3: ───────S───────
        >>> print(cirq.Circuit.zip(circuit3, circuit2, circuit1))
        0: ───H───@───────
                  │
        1: ───────X───────
        <BLANKLINE>
        2: ───X───Y───Z───
        <BLANKLINE>
        3: ───────S───────
        """
        n = max([len(c) for c in circuits], default=0)
        if isinstance(align, str):
            align = Alignment[align.upper()]
        result = cirq.Circuit()
        for k in range(n):
            try:
                if align == Alignment.LEFT:
                    moment = cirq.Moment((c[k] for c in circuits if k < len(c)))
                else:
                    moment = cirq.Moment((c[len(c) - n + k] for c in circuits if len(c) - n + k >= 0))
                result.append(moment)
            except ValueError as ex:
                raise ValueError(f'Overlapping operations between zipped circuits at moment index {k}.\n{ex}') from ex
        return result

    def concat_ragged(*circuits: 'cirq.AbstractCircuit', align: Union['cirq.Alignment', str]=Alignment.LEFT) -> 'cirq.AbstractCircuit':
        """Concatenates circuits, overlapping them if possible due to ragged edges.

        Starts with the first circuit (index 0), then iterates over the other
        circuits while folding them in. To fold two circuits together, they
        are placed one after the other and then moved inward until just before
        their operations would collide. If any of the circuits do not share
        qubits and so would not collide, the starts or ends of the circuits will
        be aligned, according to the given align parameter.

        Beware that this method is *not* associative. For example:

        >>> a, b = cirq.LineQubit.range(2)
        >>> A = cirq.Circuit(cirq.H(a))
        >>> B = cirq.Circuit(cirq.H(b))
        >>> f = cirq.Circuit.concat_ragged
        >>> f(f(A, B), A) == f(A, f(B, A))
        False
        >>> len(f(f(f(A, B), A), B)) == len(f(f(A, f(B, A)), B))
        False

        Args:
            *circuits: The circuits to concatenate.
            align: When to stop when sliding the circuits together.
                'left': Stop when the starts of the circuits align.
                'right': Stop when the ends of the circuits align.
                'first': Stop the first time either the starts or the ends align. Circuits
                    are never overlapped more than needed to align their starts (in case
                    the left circuit is smaller) or to align their ends (in case the right
                    circuit is smaller)

        Returns:
            The concatenated and overlapped circuit.
        """
        if len(circuits) == 0:
            return Circuit()
        n_acc = len(circuits[0])
        if isinstance(align, str):
            align = Alignment[align.upper()]
        pad_len = sum((len(c) for c in circuits)) - n_acc
        buffer: MutableSequence['cirq.Moment'] = [cirq.Moment()] * (pad_len * 2 + n_acc)
        offset = pad_len
        buffer[offset:offset + n_acc] = circuits[0].moments
        for k in range(1, len(circuits)):
            offset, n_acc = _concat_ragged_helper(offset, n_acc, buffer, circuits[k].moments, align)
        return cirq.Circuit(buffer[offset:offset + n_acc])

    def get_independent_qubit_sets(self) -> List[Set['cirq.Qid']]:
        """Divide circuit's qubits into independent qubit sets.

        Independent qubit sets are the qubit sets such that there are
        no entangling gates between qubits belonging to different sets.
        If this is not possible, a sequence with a single factor (the whole set of
        circuit's qubits) is returned.

        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> circuit = cirq.Circuit()
        >>> circuit.append(cirq.Moment(cirq.H(q2)))
        >>> circuit.append(cirq.Moment(cirq.CZ(q0,q1)))
        >>> circuit.append(cirq.H(q0))
        >>> print(circuit)
        0: ───────@───H───
                  │
        1: ───────@───────
        <BLANKLINE>
        2: ───H───────────
        >>> [sorted(qs) for qs in circuit.get_independent_qubit_sets()]
        [[cirq.LineQubit(0), cirq.LineQubit(1)], [cirq.LineQubit(2)]]

        Returns:
            The list of independent qubit sets.

        """
        uf = networkx.utils.UnionFind(self.all_qubits())
        for op in self.all_operations():
            if len(op.qubits) > 1:
                uf.union(*op.qubits)
        return sorted([qs for qs in uf.to_sets()], key=min)

    def factorize(self) -> Iterable[Self]:
        """Factorize circuit into a sequence of independent circuits (factors).

        Factorization is possible when the circuit's qubits can be divided
        into two or more independent qubit sets. Preserves the moments from
        the original circuit.
        If this is not possible, returns the set consisting of the single
        circuit (this one).

        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> circuit = cirq.Circuit()
        >>> circuit.append(cirq.Moment(cirq.H(q2)))
        >>> circuit.append(cirq.Moment(cirq.CZ(q0,q1)))
        >>> circuit.append(cirq.H(q0))
        >>> print(circuit)
        0: ───────@───H───
                  │
        1: ───────@───────
        <BLANKLINE>
        2: ───H───────────
        >>> for i, f in enumerate(circuit.factorize()):
        ...     print("Factor {}".format(i))
        ...     print(f)
        ...
        Factor 0
        0: ───────@───H───
                  │
        1: ───────@───────
        Factor 1
        2: ───H───────────

        Returns:
            The sequence of circuits, each including only the qubits from one
            independent qubit set.

        """
        qubit_factors = self.get_independent_qubit_sets()
        if len(qubit_factors) == 1:
            return (self,)
        return (self._from_moments((m[qubits] for m in self.moments)) for qubits in qubit_factors)

    def _control_keys_(self) -> FrozenSet['cirq.MeasurementKey']:
        controls = frozenset((k for op in self.all_operations() for k in protocols.control_keys(op)))
        return controls - protocols.measurement_key_objs(self)