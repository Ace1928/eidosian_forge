import abc
from typing import Callable, Dict, Iterator, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_function
class UnaryIterationGate(infra.GateWithRegisters):
    """Base class for defining multiplexed gates that can execute a coherent for-loop.

    Unary iteration is a coherent for loop that can be used to conditionally perform a different
    operation on a target register for every integer in the `range(l_iter, r_iter)` stored in the
    selection register.

    `cirq_ft.UnaryIterationGate` leverages the utility method `cirq_ft.unary_iteration` to provide
    a convenient API for users to define a multi-dimensional multiplexed gate that can execute
    indexed operations on a target register depending on the index value stored in a selection
    register.

    Note: Unary iteration circuits assume that the selection register stores integers only in the
    range `[l, r)` for which the corresponding unary iteration circuit should be built.

    References:
            [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
        (https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.A.
    """

    @cached_property
    @abc.abstractmethod
    def control_registers(self) -> Tuple[infra.Register, ...]:
        pass

    @cached_property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        pass

    @cached_property
    @abc.abstractmethod
    def target_registers(self) -> Tuple[infra.Register, ...]:
        pass

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature([*self.control_registers, *self.selection_registers, *self.target_registers])

    @cached_property
    def extra_registers(self) -> Tuple[infra.Register, ...]:
        return ()

    @abc.abstractmethod
    def nth_operation(self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs) -> cirq.OP_TREE:
        """Apply nth operation on the target signature when selection signature store `n`.

        The `UnaryIterationGate` class is a mixin that represents a coherent for-loop over
        different indices (i.e. selection signature). This method denotes the "body" of the
        for-loop, which is executed `self.selection_registers.total_iteration_size` times and each
        iteration represents a unique combination of values stored in selection signature. For each
        call, the method should return the operations that should be applied to the target
        signature, given the values stored in selection signature.

        The derived classes should specify the following arguments as `**kwargs`:
            1) `control: cirq.Qid`: A qubit which can be used as a control to selectively
            apply operations when selection register stores specific value.
            2) Register names in `self.selection_registers`: Each argument corresponds to
            a selection register and represents the integer value stored in the register.
            3) Register names in `self.target_registers`: Each argument corresponds to a target
            register and represents the sequence of qubits that represent the target register.
            4) Register names in `self.extra_regs`: Each argument corresponds to an extra
            register and represents the sequence of qubits that represent the extra register.
        """

    def decompose_zero_selection(self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
        """Specify decomposition of the gate when selection register is empty

        By default, if the selection register is empty, the decomposition will raise a
        `NotImplementedError`. The derived classes can override this method and specify
        a custom decomposition that should be used if the selection register is empty,
        i.e. `infra.total_bits(self.selection_registers) == 0`.

        The derived classes should specify the following arguments as `**kwargs`:
            1) Register names in `self.control_registers`: Each argument corresponds to a
            control register and represents sequence of qubits that represent the control register.
            2) Register names in `self.target_registers`: Each argument corresponds to a target
            register and represents the sequence of qubits that represent the target register.
            3) Register names in `self.extra_regs`: Each argument corresponds to an extra
            register and represents the sequence of qubits that represent the extra register.
        """
        raise NotImplementedError('Selection register must not be empty.')

    def _break_early(self, selection_index_prefix: Tuple[int, ...], l: int, r: int) -> bool:
        """Derived classes should override this method to specify an early termination condition.

        For each internal node of the unary iteration segment tree, `break_early(l, r)` is called
        to evaluate whether the unary iteration should not recurse in the subtree of the node
        representing range `[l, r)`. If True, the internal node is considered equivalent to a leaf
        node and thus, `self.nth_operation` will be called for only integer `l` in the range [l, r).

        When the `UnaryIteration` class is constructed using multiple selection signature, i.e. we
        wish to perform nested coherent for-loops, a unary iteration segment tree is constructed
        corresponding to each nested coherent for-loop. For every such unary iteration segment tree,
        the `_break_early` condition is checked by passing the `selection_index_prefix` tuple.

        Args:
            selection_index_prefix: To evaluate the early breaking condition for the i'th nested
                for-loop, the `selection_index_prefix` contains `i-1` integers corresponding to
                the loop variable values for the first `i-1` nested loops.
            l: Beginning of range `[l, r)` for internal node of unary iteration segment tree.
            r: End (exclusive) of range `[l, r)` for internal node of unary iteration segment tree.

        Returns:
            True of the `len(selection_index_prefix)`'th unary iteration should terminate early for
            the given parameters.
        """
        return False

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
        if infra.total_bits(self.selection_registers) == 0 or self._break_early((), 0, self.selection_registers[0].iteration_length):
            return self.decompose_zero_selection(context=context, **quregs)
        num_loops = len(self.selection_registers)
        target_regs = {reg.name: quregs[reg.name] for reg in self.target_registers}
        extra_regs = {reg.name: quregs[reg.name] for reg in self.extra_registers}

        def unary_iteration_loops(nested_depth: int, selection_reg_name_to_val: Dict[str, int], controls: Sequence[cirq.Qid]) -> Iterator[cirq.OP_TREE]:
            """Recursively write any number of nested coherent for-loops using unary iteration.

            This helper method is useful to write `num_loops` number of nested coherent for-loops by
            recursively calling this method `num_loops` times. The ith recursive call of this method
            has `nested_depth=i` and represents the body of ith nested for-loop.

            Args:
                nested_depth: Integer between `[0, num_loops]` representing the nest-level of
                    for-loop for which this method implements the body.
                selection_reg_name_to_val: A dictionary containing `nested_depth` elements mapping
                    the selection integer names (i.e. loop variables) to corresponding values;
                    for each of the `nested_depth` parent for-loops written before.
                controls: Control qubits that should be used to conditionally activate the body of
                    this for-loop.

            Returns:
                `cirq.OP_TREE` implementing `num_loops` nested coherent for-loops, with operations
                returned by `self.nth_operation` applied conditionally to the target register based
                on values of selection signature.
            """
            if nested_depth == num_loops:
                yield self.nth_operation(context=context, control=controls[0], **selection_reg_name_to_val, **target_regs, **extra_regs)
                return
            ops: List[cirq.Operation] = []
            selection_index_prefix = tuple(selection_reg_name_to_val.values())
            ith_for_loop = unary_iteration(l_iter=0, r_iter=self.selection_registers[nested_depth].iteration_length, flanking_ops=ops, controls=controls, selection=[*quregs[self.selection_registers[nested_depth].name]], qubit_manager=context.qubit_manager, break_early=lambda l, r: self._break_early(selection_index_prefix, l, r))
            for op_tree, control_qid, n in ith_for_loop:
                yield op_tree
                selection_reg_name_to_val[self.selection_registers[nested_depth].name] = n
                yield from unary_iteration_loops(nested_depth + 1, selection_reg_name_to_val, (control_qid,))
                selection_reg_name_to_val.pop(self.selection_registers[nested_depth].name)
            yield ops
        return unary_iteration_loops(0, {}, infra.merge_qubits(self.control_registers, **quregs))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Basic circuit diagram.

        Descendants are encouraged to override this with more descriptive
        circuit diagram information.
        """
        wire_symbols = ['@'] * infra.total_bits(self.control_registers)
        wire_symbols += ['In'] * infra.total_bits(self.selection_registers)
        wire_symbols += [self.__class__.__name__] * infra.total_bits(self.target_registers)
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)