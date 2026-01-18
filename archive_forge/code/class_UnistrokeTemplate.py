import pickle
import base64
import zlib
from re import match as re_match
from collections import deque
from math import sqrt, pi, radians, acos, atan, atan2, pow, floor
from math import sin as math_sin, cos as math_cos
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.properties import ListProperty
from kivy.compat import PY2
from io import BytesIO
class UnistrokeTemplate(object):
    """Represents a (uni)stroke path as a list of Vectors. Normally, this class
    is instantiated by MultistrokeGesture and not by the programmer directly.
    However, it is possible to manually compose UnistrokeTemplate objects.

    :Arguments:
        `name`
            Identifies the name of the gesture. This is normally inherited from
            the parent MultistrokeGesture object when a template is generated.
        `points`
            A list of points that represents a unistroke path. This is normally
            one of the possible stroke order permutations from a
            MultistrokeGesture.
        `numpoints`
            The number of points this template should (ideally) be resampled to
            before the matching process. The default is 16, but you can use a
            template-specific settings if that improves results.
        `orientation_sensitive`
            Determines if this template is orientation sensitive (True) or
            fully rotation invariant (False). The default is True.

    .. Note::
        You will get an exception if you set a skip-flag and then attempt to
        retrieve those vectors.
    """

    def __init__(self, name, points=None, **kwargs):
        self.name = name
        self.numpoints = kwargs.get('numpoints', 16)
        self.orientation_sens = kwargs.get('orientation_sensitive', True)
        self.db = {}
        self.points = []
        if points is not None:
            self.points = points

    def add_point(self, p):
        """Add a point to the unistroke/path. This invalidates all previously
        computed vectors."""
        self.points.append(p)
        self.db = {}

    def _get_db_key(self, key, numpoints=None):
        n = numpoints and numpoints or self.numpoints
        if n not in self.db:
            self.prepare(n)
        return self.db[n][key]

    def get_start_unit_vector(self, numpoints=None):
        return self._get_db_key('startvector', numpoints)

    def get_vector(self, numpoints=None):
        return self._get_db_key('vector', numpoints)

    def get_points(self, numpoints=None):
        return self._get_db_key('points', numpoints)

    def prepare(self, numpoints=None):
        """This function prepares the UnistrokeTemplate for matching given a
        target number of points (for resample). 16 is optimal."""
        if not self.points:
            raise MultistrokeError('prepare() called without self.points')
        n = numpoints or self.numpoints
        if not n or n < 2:
            raise MultistrokeError('prepare() called with invalid numpoints')
        p = resample(self.points, n)
        radians = indicative_angle(p)
        p = rotate_by(p, -radians)
        p = scale_dim(p, SQUARESIZE, ONEDTHRESHOLD)
        if self.orientation_sens:
            p = rotate_by(p, +radians)
        p = translate_to(p, ORIGIN)
        self.db[n] = {'startvector': start_unit_vector(p, n / 8), 'vector': vectorize(p, self.orientation_sens)}