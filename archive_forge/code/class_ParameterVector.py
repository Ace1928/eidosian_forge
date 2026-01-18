from uuid import uuid4, UUID
from .parameter import Parameter
class ParameterVector:
    """ParameterVector class to quickly generate lists of parameters."""
    __slots__ = ('_name', '_params', '_size', '_root_uuid')

    def __init__(self, name, length=0):
        self._name = name
        self._size = length
        self._root_uuid = uuid4()
        root_uuid_int = self._root_uuid.int
        self._params = [ParameterVectorElement(self, i, UUID(int=root_uuid_int + i)) for i in range(length)]

    @property
    def name(self):
        """Returns the name of the ParameterVector."""
        return self._name

    @property
    def params(self):
        """Returns the list of parameters in the ParameterVector."""
        return self._params

    def index(self, value):
        """Returns first index of value."""
        return self._params.index(value)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._size)
            return self.params[start:stop:step]
        if key > self._size:
            raise IndexError(f'Index out of range: {key} > {self._size}')
        return self.params[key]

    def __iter__(self):
        return iter(self.params[:self._size])

    def __len__(self):
        return self._size

    def __str__(self):
        return f'{self.name}, {[str(item) for item in self.params[:self._size]]}'

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, length={len(self)})'

    def resize(self, length):
        """Resize the parameter vector.

        If necessary, new elements are generated. If length is smaller than before, the
        previous elements are cached and not re-generated if the vector is enlarged again.
        This is to ensure that the parameter instances do not change.
        """
        if length > len(self._params):
            root_uuid_int = self._root_uuid.int
            self._params.extend([ParameterVectorElement(self, i, UUID(int=root_uuid_int + i)) for i in range(len(self._params), length)])
        self._size = length