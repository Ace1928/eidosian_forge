import abc
import copy
from neutron_lib import exceptions
class FilterObj(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def filter(self, column):
        pass