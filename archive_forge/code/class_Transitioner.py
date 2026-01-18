from itertools import chain
class Transitioner(object):
    """
    The combination of a current state and an L{Automaton}.
    """

    def __init__(self, automaton, initialState):
        self._automaton = automaton
        self._state = initialState
        self._tracer = None

    def setTrace(self, tracer):
        self._tracer = tracer

    def transition(self, inputSymbol):
        """
        Transition between states, returning any outputs.
        """
        outState, outputSymbols = self._automaton.outputForInput(self._state, inputSymbol)
        outTracer = None
        if self._tracer:
            outTracer = self._tracer(self._state._name(), inputSymbol._name(), outState._name())
        self._state = outState
        return (outputSymbols, outTracer)