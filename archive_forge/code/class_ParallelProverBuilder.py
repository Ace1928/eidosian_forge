import threading
import time
from abc import ABCMeta, abstractmethod
class ParallelProverBuilder(Prover, ModelBuilder):
    """
    This class stores both a prover and a model builder and when either
    prove() or build_model() is called, then both theorem tools are run in
    parallel.  Whichever finishes first, the prover or the model builder, is the
    result that will be used.
    """

    def __init__(self, prover, modelbuilder):
        self._prover = prover
        self._modelbuilder = modelbuilder

    def _prove(self, goal=None, assumptions=None, verbose=False):
        return (self._run(goal, assumptions, verbose), '')

    def _build_model(self, goal=None, assumptions=None, verbose=False):
        return (not self._run(goal, assumptions, verbose), '')

    def _run(self, goal, assumptions, verbose):
        tp_thread = TheoremToolThread(lambda: self._prover.prove(goal, assumptions, verbose), verbose, 'TP')
        mb_thread = TheoremToolThread(lambda: self._modelbuilder.build_model(goal, assumptions, verbose), verbose, 'MB')
        tp_thread.start()
        mb_thread.start()
        while tp_thread.is_alive() and mb_thread.is_alive():
            pass
        if tp_thread.result is not None:
            return tp_thread.result
        elif mb_thread.result is not None:
            return not mb_thread.result
        else:
            return None