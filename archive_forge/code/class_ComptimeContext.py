import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
class ComptimeContext:
    """
    This context class provides access to a public API for Dynamo's internals.
    If there is something here you would find useful that is missing, please
    file a feature request at https://github.com/pytorch/pytorch/
    """

    def __init__(self, tx):
        self.__tx = tx

    def get_local(self, name: str, *, stacklevel=0) -> ComptimeVar:
        """
        Retrieve the compile-time known information about a local.
        """
        tx = self.__get_tx(stacklevel)
        return ComptimeVar(tx.symbolic_locals[name])

    def graph_break(self, msg='ComptimeContext.graph_break'):
        """
        Manually trigger a graph break
        """
        unimplemented(msg)

    def graph(self):
        """
        Retrieve the partially constructed FX graph that would be
        passed to the user compiler after compilation.
        """
        return self.__tx.output.graph

    def print_graph(self, *, verbose=True, file=None):
        """
        Print the partially constructed FX graph that would be passed
        to the user compiler after compilation.
        """
        print(self.__tx.output.graph.python_code('self', verbose=verbose).src, file=file)

    def parent(self):
        return ComptimeContext(self.__tx.parent)

    def __get_tx(self, stacklevel):
        tx = self.__tx
        for _ in range(stacklevel):
            tx = tx.parent
        return tx

    def print_disas(self, *, file=None, stacklevel=0):
        """
        Print the current series of opcodes being executed (not including
        parent frames), including where you are in the particular opcode
        stream.
        """
        tx = self.__get_tx(stacklevel)
        print(dis.Bytecode(tx.f_code, current_offset=tx.instructions[tx.instruction_pointer].offset).dis(), file=file)

    def print_value_stack(self, *, file=None, stacklevel=0):
        """
        Print the current Python value stack.  Note that this is NOT the same
        as the traceback; use print_bt() to print that.  Note that at
        stacklevel=0, this will typically be empty, as comptime cannot
        currently be used in an expression context where there would be
        intermediates on the stack.  If you would find this useful, please
        file a bug at https://github.com/pytorch/pytorch/

        NB: Stack grows downwards in our print
        """
        tx = self.__get_tx(stacklevel)
        for s in tx.stack:
            print(f'- {s}', file=file)

    def print_locals(self, *, file=None, stacklevel=0):
        """
        Print all of the locals available in the current context.
        By default this view is very limited; you can get more information
        about any individual local using get_local().
        """
        tx = self.__get_tx(stacklevel)
        for k, v in tx.symbolic_locals.items():
            print(f'{k} = {v}', file=file)

    def print_bt(self, *, file=None, stacklevel=0):
        """
        Print the user code backtrace, starting at the beginning of the
        frame Dynamo started evaluating.  Note that this MAY NOT go all
        the way to the torch.compile invocation, as we may have done
        a graph break and are compiling an intermediate frame as the
        starting point.  If you think the other behavior would be better,
        file a bug at https://github.com/pytorch/pytorch/
        """
        stack = []
        tx = self.__get_tx(stacklevel)
        while tx is not None:
            stack.append(tx.frame_summary())
            tx = getattr(tx, 'parent', None)
        print(''.join(traceback.StackSummary.from_list(reversed(stack)).format()), file=file)

    def print_guards(self, *, file=None):
        """
        Print the currently installed guards for the Dynamo context.
        This does NOT include guards associated with variables that
        may or may not be installed in the future if those variables
        are used.
        """
        print('\n'.join((f'{repr(guard)}' for guard in sorted(self.__tx.output.guards))), file=file)

    def _i_will_not_complain_if_bc_breaks_InstructionTranslator(self):
        """
        Returns the internal data structure InstructionTranslator that Dynamo
        uses to track state of symbolic evaluation.  There are no BC
        guarantees on this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if
        you rely on it.
        """
        return self.__tx