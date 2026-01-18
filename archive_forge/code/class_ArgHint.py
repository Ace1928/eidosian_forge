import abc
from numba.core.typing.typeof import typeof, Purpose
class ArgHint(metaclass=abc.ABCMeta):

    def __init__(self, value):
        self.value = value

    @abc.abstractmethod
    def to_device(self, retr, stream=0):
        """
        :param stream: a stream to use when copying data
        :param retr:
            a list of clean-up work to do after the kernel's been run.
            Append 0-arg lambdas to it!
        :return: a value (usually an `DeviceNDArray`) to be passed to
            the kernel
        """
        pass

    @property
    def _numba_type_(self):
        return typeof(self.value, Purpose.argument)