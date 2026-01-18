import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
class Rectangle:
    """Hyperrectangle class.

    Represents a Cartesian product of intervals.
    """

    def __init__(self, maxes, mins):
        """Construct a hyperrectangle."""
        self.maxes = np.maximum(maxes, mins).astype(float)
        self.mins = np.minimum(maxes, mins).astype(float)
        self.m, = self.maxes.shape

    def __repr__(self):
        return '<Rectangle %s>' % list(zip(self.mins, self.maxes))

    def volume(self):
        """Total volume."""
        return np.prod(self.maxes - self.mins)

    def split(self, d, split):
        """Produce two hyperrectangles by splitting.

        In general, if you need to compute maximum and minimum
        distances to the children, it can be done more efficiently
        by updating the maximum and minimum distances to the parent.

        Parameters
        ----------
        d : int
            Axis to split hyperrectangle along.
        split : float
            Position along axis `d` to split at.

        """
        mid = np.copy(self.maxes)
        mid[d] = split
        less = Rectangle(self.mins, mid)
        mid = np.copy(self.mins)
        mid[d] = split
        greater = Rectangle(mid, self.maxes)
        return (less, greater)

    def min_distance_point(self, x, p=2.0):
        """
        Return the minimum distance between input and points in the
        hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input.
        p : float, optional
            Input.

        """
        return minkowski_distance(0, np.maximum(0, np.maximum(self.mins - x, x - self.maxes)), p)

    def max_distance_point(self, x, p=2.0):
        """
        Return the maximum distance between input and points in the hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input array.
        p : float, optional
            Input.

        """
        return minkowski_distance(0, np.maximum(self.maxes - x, x - self.mins), p)

    def min_distance_rectangle(self, other, p=2.0):
        """
        Compute the minimum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float
            Input.

        """
        return minkowski_distance(0, np.maximum(0, np.maximum(self.mins - other.maxes, other.mins - self.maxes)), p)

    def max_distance_rectangle(self, other, p=2.0):
        """
        Compute the maximum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float, optional
            Input.

        """
        return minkowski_distance(0, np.maximum(self.maxes - other.mins, other.maxes - self.mins), p)