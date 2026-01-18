import abc
from numba.core.typing.typeof import typeof, Purpose
class InOut(ArgHint):

    def to_device(self, retr, stream=0):
        from .cudadrv.devicearray import auto_device
        devary, conv = auto_device(self.value, stream=stream)
        if conv:
            retr.append(lambda: devary.copy_to_host(self.value, stream=stream))
        return devary