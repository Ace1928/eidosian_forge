from numba import types
from numba.cuda.stubs import _vector_type_stubs
class SimulatedVectorType:
    attributes = ['x', 'y', 'z', 'w']

    def __init__(self, *args):
        args_flattened = []
        for arg in args:
            if isinstance(arg, SimulatedVectorType):
                args_flattened += arg.as_list()
            else:
                args_flattened.append(arg)
        self._attrs = self.attributes[:len(args_flattened)]
        if not self.num_elements == len(args_flattened):
            raise TypeError(f'{self.name} expects {self.num_elements} elements, got {len(args_flattened)}')
        for arg, attr in zip(args_flattened, self._attrs):
            setattr(self, attr, arg)

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def num_elements(self):
        raise NotImplementedError()

    def as_list(self):
        return [getattr(self, attr) for attr in self._attrs]