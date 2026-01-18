import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class AnyOneAbc(metaclass=ABCMetaImplementAnyOneOf):

    def _method1_with_2(self):
        return '1-2 ' + self.method2()

    def _method1_with_3(self):
        return '1-3 ' + self.method3()

    def _method2_with_1(self):
        return '2-1 ' + self.method1()

    def _method2_with_3(self):
        return '2-3 ' + self.method3()

    def _method3_with_1(self):
        return '3-1 ' + self.method1()

    def _method3_with_2(self):
        return '3-2 ' + self.method2()

    @alternative(requires='method2', implementation=_method1_with_2)
    @alternative(requires='method3', implementation=_method1_with_3)
    def method1(self):
        """Method1."""

    @alternative(requires='method1', implementation=_method2_with_1)
    @alternative(requires='method3', implementation=_method2_with_3)
    def method2(self):
        """Method2."""

    @alternative(requires='method1', implementation=_method3_with_1)
    @alternative(requires='method2', implementation=_method3_with_2)
    def method3(self):
        """Method3."""