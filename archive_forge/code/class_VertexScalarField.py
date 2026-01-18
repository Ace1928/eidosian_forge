import collections
from abc import ABC, abstractmethod
import numpy as np
from scipy._lib._util import MapWrapper
class VertexScalarField(VertexBase):
    """
    Add homology properties of a scalar field f: R^n --> R associated with
    the geometry built from the VertexBase class
    """

    def __init__(self, x, field=None, nn=None, index=None, field_args=(), g_cons=None, g_cons_args=()):
        """
        Parameters
        ----------
        x : tuple,
            vector of vertex coordinates
        field : callable, optional
            a scalar field f: R^n --> R associated with the geometry
        nn : list, optional
            list of nearest neighbours
        index : int, optional
            index of the vertex
        field_args : tuple, optional
            additional arguments to be passed to field
        g_cons : callable, optional
            constraints on the vertex
        g_cons_args : tuple, optional
            additional arguments to be passed to g_cons

        """
        super().__init__(x, nn=nn, index=index)
        self.check_min = True
        self.check_max = True

    def connect(self, v):
        """Connects self to another vertex object v.

        Parameters
        ----------
        v : VertexBase or VertexScalarField object
        """
        if v is not self and v not in self.nn:
            self.nn.add(v)
            v.nn.add(self)
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)
            self.check_min = True
            self.check_max = True
            v.check_min = True
            v.check_max = True

    def minimiser(self):
        """Check whether this vertex is strictly less than all its
           neighbours"""
        if self.check_min:
            self._min = all((self.f < v.f for v in self.nn))
            self.check_min = False
        return self._min

    def maximiser(self):
        """
        Check whether this vertex is strictly greater than all its
        neighbours.
        """
        if self.check_max:
            self._max = all((self.f > v.f for v in self.nn))
            self.check_max = False
        return self._max