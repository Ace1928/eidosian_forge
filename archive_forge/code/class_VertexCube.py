import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
class VertexCube(VertexBase):
    """Vertex class to be used for a pure simplicial complex with no associated
    differential geometry (single level domain that exists in R^n)"""

    def __init__(self, x, nn=None, index=None):
        super().__init__(x, nn=nn, index=index)

    def connect(self, v):
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)