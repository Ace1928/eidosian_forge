import threading
import time
from abc import ABCMeta, abstractmethod
class BaseModelBuilderCommand(BaseTheoremToolCommand, ModelBuilderCommand):
    """
    This class holds a ``ModelBuilder``, a goal, and a list of assumptions.  When
    build_model() is called, the ``ModelBuilder`` is executed with the goal and
    assumptions.
    """

    def __init__(self, modelbuilder, goal=None, assumptions=None):
        """
        :param modelbuilder: The theorem tool to execute with the assumptions
        :type modelbuilder: ModelBuilder
        :see: ``BaseTheoremToolCommand``
        """
        self._modelbuilder = modelbuilder
        'The theorem tool to execute with the assumptions'
        BaseTheoremToolCommand.__init__(self, goal, assumptions)
        self._model = None

    def build_model(self, verbose=False):
        """
        Attempt to build a model.  Store the result to prevent unnecessary
        re-building.
        """
        if self._result is None:
            self._result, self._model = self._modelbuilder._build_model(self.goal(), self.assumptions(), verbose)
        return self._result

    def model(self, format=None):
        """
        Return a string representation of the model

        :param simplify: bool simplify the proof?
        :return: str
        """
        if self._result is None:
            raise LookupError('You have to call build_model() first to get a model!')
        else:
            return self._decorate_model(self._model, format)

    def _decorate_model(self, valuation_str, format=None):
        """
        :param valuation_str: str with the model builder's output
        :param format: str indicating the format for displaying
        :return: str
        """
        return valuation_str

    def get_model_builder(self):
        return self._modelbuilder