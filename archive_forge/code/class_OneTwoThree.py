import dill
import abc
from abc import ABC
import warnings
from types import FunctionType
class OneTwoThree(ABC):

    @abc.abstractmethod
    def foo(self):
        """A method"""
        pass

    @property
    @abc.abstractmethod
    def bar(self):
        """Property getter"""
        pass

    @bar.setter
    @abc.abstractmethod
    def bar(self, value):
        """Property setter"""
        pass

    @classmethod
    @abc.abstractmethod
    def cfoo(cls):
        """Class method"""
        pass

    @staticmethod
    @abc.abstractmethod
    def sfoo():
        """Static method"""
        pass