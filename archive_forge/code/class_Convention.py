import abc
import re
class Convention(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def convert_function_name(self, name):
        pass

    @abc.abstractmethod
    def convert_parameter_name(self, name):
        pass