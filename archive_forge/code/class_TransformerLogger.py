import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
class TransformerLogger:
    """Base Class for transformer logging infrastructure. Defaults to text-based logging.

    The logger implementation should be stateful, s.t.:
        - Each call to `register_initial` registers a new transformer stage and initial circuit.
        - Each subsequent call to `log` should store additional logs corresponding to the stage.
        - Each call to `register_final` should register the end of the currently active stage.

    The logger assumes that
        - Transformers are run sequentially.
        - Nested transformers are allowed, in which case the behavior would be similar to a
          doing a depth first search on the graph of transformers -- i.e. the top level transformer
          would end (i.e. receive a `register_final` call) once all nested transformers (i.e. all
          `register_initial` calls received while the top level transformer was active) have
          finished (i.e. corresponding `register_final` calls have also been received).
        - This behavior can be simulated by maintaining a stack of currently active stages and
          adding data from `log` calls to the stage at the top of the stack.

    The `LogLevel`s can be used to control the input processing and output resolution of the logs.
    """

    def __init__(self) -> None:
        """Initializes TransformerLogger."""
        self._curr_id: int = 0
        self._logs: List[_LoggerNode] = []
        self._stack: List[int] = []

    def register_initial(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        """Register the beginning of a new transformer stage.

        Args:
            circuit: Input circuit to the new transformer stage.
            transformer_name: Name of the new transformer stage.
        """
        if self._stack:
            self._logs[self._stack[-1]].nested_loggers.append(self._curr_id)
        self._logs.append(_LoggerNode(self._curr_id, transformer_name, circuit, circuit))
        self._stack.append(self._curr_id)
        self._curr_id += 1

    def log(self, *args: str, level: LogLevel=LogLevel.INFO) -> None:
        """Log additional metadata corresponding to the currently active transformer stage.

        Args:
            *args: The additional metadata to log.
            level: Logging level to control the amount of metadata that gets put into the context.

        Raises:
            ValueError: If there's no active transformer on the stack.
        """
        if len(self._stack) == 0:
            raise ValueError('No active transformer found.')
        self._logs[self._stack[-1]].logs.append((level, args))

    def register_final(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        """Register the end of the currently active transformer stage.

        Args:
            circuit: Final transformed output circuit from the transformer stage.
            transformer_name: Name of the (currently active) transformer stage which ends.

        Raises:
            ValueError: If `transformer_name` is different from currently active transformer name.
        """
        tid = self._stack.pop()
        if self._logs[tid].transformer_name != transformer_name:
            raise ValueError(f'Expected `register_final` call for currently active transformer {self._logs[tid].transformer_name}.')
        self._logs[tid].final_circuit = circuit

    def show(self, level: LogLevel=LogLevel.INFO) -> None:
        """Show the stored logs >= level in the desired format.

        Args:
            level: The logging level to filter the logs with. The method shows all logs with a
            `LogLevel` >= `level`.
        """

        def print_log(log: _LoggerNode, pad=''):
            print(pad, f'Transformer-{1 + log.transformer_id}: {log.transformer_name}', sep='')
            print(pad, 'Initial Circuit:', sep='')
            print(textwrap.indent(str(log.initial_circuit), pad), '\n', sep='')
            for log_level, log_text in log.logs:
                if log_level.value >= level.value:
                    print(pad, log_level, *log_text)
            print('\n', pad, 'Final Circuit:', sep='')
            print(textwrap.indent(str(log.final_circuit), pad))
            print('----------------------------------------')
        done = [0] * self._curr_id
        for i in range(self._curr_id):
            stack = [(i, '')] if not done[i] else []
            while len(stack) > 0:
                log_id, pad = stack.pop()
                print_log(self._logs[log_id], pad)
                done[log_id] = True
                for child_id in self._logs[log_id].nested_loggers[::-1]:
                    stack.append((child_id, pad + ' ' * 4))