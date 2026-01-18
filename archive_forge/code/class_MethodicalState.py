import collections
from functools import wraps
from itertools import count
from inspect import getfullargspec as getArgsSpec
import attr
from ._core import Transitioner, Automaton
from ._introspection import preserveName
@attr.s(frozen=True)
class MethodicalState(object):
    """
    A state for a L{MethodicalMachine}.
    """
    machine = attr.ib(repr=False)
    method = attr.ib()
    serialized = attr.ib(repr=False)

    def upon(self, input, enter=None, outputs=None, collector=list):
        """
        Declare a state transition within the :class:`automat.MethodicalMachine`
        associated with this :class:`automat.MethodicalState`:
        upon the receipt of the `input`, enter the `state`,
        emitting each output in `outputs`.

        :param MethodicalInput input: The input triggering a state transition.
        :param MethodicalState enter: The resulting state.
        :param Iterable[MethodicalOutput] outputs: The outputs to be triggered
            as a result of the declared state transition.
        :param Callable collector: The function to be used when collecting
            output return values.

        :raises TypeError: if any of the `outputs` signatures do not match
            the `inputs` signature.
        :raises ValueError: if the state transition from `self` via `input`
            has already been defined.
        """
        if enter is None:
            enter = self
        if outputs is None:
            outputs = []
        inputArgs = _getArgNames(input.argSpec)
        for output in outputs:
            outputArgs = _getArgNames(output.argSpec)
            if not outputArgs.issubset(inputArgs):
                raise TypeError('method {input} signature {inputSignature} does not match output {output} signature {outputSignature}'.format(input=input.method.__name__, output=output.method.__name__, inputSignature=getArgsSpec(input.method), outputSignature=getArgsSpec(output.method)))
        self.machine._oneTransition(self, input, enter, outputs, collector)

    def _name(self):
        return self.method.__name__