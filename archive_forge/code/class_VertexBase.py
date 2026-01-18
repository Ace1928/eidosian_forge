import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
class VertexBase(ABC):
    """
    Base class for a vertex.
    """

    def __init__(self, x, nn=None, index=None):
        """
        Initiation of a vertex object.

        Parameters
        ----------
        x : tuple or vector
            The geometric location (domain).
        nn : list, optional
            Nearest neighbour list.
        index : int, optional
            Index of vertex.
        """
        self.x = x
        self.hash = hash(self.x)
        if nn is not None:
            self.nn = set(nn)
        else:
            self.nn = set()
        self.index = index

    def __hash__(self):
        return self.hash

    def __getattr__(self, item):
        if item not in ['x_a']:
            raise AttributeError(f"{type(self)} object has no attribute '{item}'")
        if item == 'x_a':
            self.x_a = np.array(self.x)
            return self.x_a

    @abstractmethod
    def connect(self, v):
        raise NotImplementedError('This method is only implemented with an associated child of the base class.')

    @abstractmethod
    def disconnect(self, v):
        raise NotImplementedError('This method is only implemented with an associated child of the base class.')

    def star(self):
        """Returns the star domain ``st(v)`` of the vertex.

        Parameters
        ----------
        v :
            The vertex ``v`` in ``st(v)``

        Returns
        -------
        st : set
            A set containing all the vertices in ``st(v)``
        """
        self.st = self.nn
        self.st.add(self)
        return self.st