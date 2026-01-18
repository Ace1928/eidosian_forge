import threading
import time
from abc import ABCMeta, abstractmethod
class ModelBuilderCommandDecorator(TheoremToolCommandDecorator, ModelBuilderCommand):
    """
    A base decorator for the ``ModelBuilderCommand`` class from which other
    prover command decorators can extend.
    """

    def __init__(self, modelBuilderCommand):
        """
        :param modelBuilderCommand: ``ModelBuilderCommand`` to decorate
        """
        TheoremToolCommandDecorator.__init__(self, modelBuilderCommand)
        self._model = None

    def build_model(self, verbose=False):
        """
        Attempt to build a model.  Store the result to prevent unnecessary
        re-building.
        """
        if self._result is None:
            modelbuilder = self.get_model_builder()
            self._result, self._model = modelbuilder._build_model(self.goal(), self.assumptions(), verbose)
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
        Modify and return the proof string
        :param valuation_str: str with the model builder's output
        :param format: str indicating the format for displaying
        :return: str
        """
        return self._command._decorate_model(valuation_str, format)

    def get_model_builder(self):
        return self._command.get_prover()