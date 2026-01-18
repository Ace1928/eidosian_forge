from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
class Vec2:
    __slots__ = ('x', 'y')
    'A two-dimensional vector represented as an X Y coordinate pair.'

    def __init__(self, x: float=0.0, y: float=0.0) -> None:
        self.x = x
        self.y = y

    def __iter__(self) -> _Iterator[float]:
        yield self.x
        yield self.y

    @_typing.overload
    def __getitem__(self, item: int) -> float:
        ...

    @_typing.overload
    def __getitem__(self, item: slice) -> tuple[float, ...]:
        ...

    def __getitem__(self, item):
        return (self.x, self.y)[item]

    def __setitem__(self, key, value):
        if type(key) is slice:
            for i, attr in enumerate(['x', 'y'][key]):
                setattr(self, attr, value[i])
        else:
            setattr(self, ['x', 'y'][key], value)

    def __len__(self) -> int:
        return 2

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)

    def __floordiv__(self, scalar: float) -> Vec2:
        return Vec2(self.x // scalar, self.y // scalar)

    def __abs__(self) -> float:
        return _math.sqrt(self.x ** 2 + self.y ** 2)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def __round__(self, ndigits: int | None=None) -> Vec2:
        return Vec2(*(round(v, ndigits) for v in self))

    def __radd__(self, other: Vec2 | int) -> Vec2:
        """Reverse add. Required for functionality with sum()
        """
        if other == 0:
            return self
        else:
            return self.__add__(_typing.cast(Vec2, other))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vec2) and self.x == other.x and (self.y == other.y)

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, Vec2) or self.x != other.x or self.y != other.y

    @staticmethod
    def from_polar(mag: float, angle: float) -> Vec2:
        """Create a new vector from the given polar coordinates.

        :parameters:
            `mag`   : int or float :
                The magnitude of the vector.
            `angle` : int or float :
                The angle of the vector in radians.
        """
        return Vec2(mag * _math.cos(angle), mag * _math.sin(angle))

    def from_magnitude(self, magnitude: float) -> Vec2:
        """Create a new Vector of the given magnitude by normalizing,
        then scaling the vector. The heading remains unchanged.

        :parameters:
            `magnitude` : int or float :
                The magnitude of the new vector.
        """
        return self.normalize() * magnitude

    def from_heading(self, heading: float) -> Vec2:
        """Create a new vector of the same magnitude with the given heading. I.e. Rotate the vector to the heading.

        :parameters:
            `heading` : int or float :
                The angle of the new vector in radians.
        """
        mag = self.__abs__()
        return Vec2(mag * _math.cos(heading), mag * _math.sin(heading))

    @property
    def heading(self) -> float:
        """The angle of the vector in radians."""
        return _math.atan2(self.y, self.x)

    @property
    def mag(self) -> float:
        """The magnitude, or length of the vector. The distance between the coordinates and the origin.

        Alias of abs(self).
        """
        return self.__abs__()

    def limit(self, maximum: float) -> Vec2:
        """Limit the magnitude of the vector to passed maximum value."""
        if self.x ** 2 + self.y ** 2 > maximum * maximum:
            return self.from_magnitude(maximum)
        return self

    def lerp(self, other: Vec2, alpha: float) -> Vec2:
        """Create a new Vec2 linearly interpolated between this vector and another Vec2.

        :parameters:
            `other`  : Vec2 :
                The vector to linearly interpolate with.
            `alpha` : float or int :
                The amount of interpolation.
                Some value between 0.0 (this vector) and 1.0 (other vector).
                0.5 is halfway inbetween.
        """
        return Vec2(self.x + alpha * (other.x - self.x), self.y + alpha * (other.y - self.y))

    def reflect(self, normal: Vec2) -> Vec2:
        """Create a new Vec2 reflected (ricochet) from the given normal."""
        return self - normal * 2 * normal.dot(self)

    def rotate(self, angle: float) -> Vec2:
        """Create a new Vector rotated by the angle. The magnitude remains unchanged."""
        s = _math.sin(angle)
        c = _math.cos(angle)
        return Vec2(c * self.x - s * self.y, s * self.x + c * self.y)

    def distance(self, other: Vec2) -> float:
        """Calculate the distance between this vector and another 2D vector."""
        return _math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)

    def normalize(self) -> Vec2:
        """Normalize the vector to have a magnitude of 1. i.e. make it a unit vector."""
        d = self.__abs__()
        if d:
            return Vec2(self.x / d, self.y / d)
        return self

    def clamp(self, min_val: float, max_val: float) -> Vec2:
        """Restrict the value of the X and Y components of the vector to be within the given values."""
        return Vec2(clamp(self.x, min_val, max_val), clamp(self.y, min_val, max_val))

    def dot(self, other: Vec2) -> float:
        """Calculate the dot product of this vector and another 2D vector."""
        return self.x * other.x + self.y * other.y

    def __getattr__(self, attrs: str) -> Vec2 | Vec3 | Vec4:
        try:
            vec_class = {2: Vec2, 3: Vec3, 4: Vec4}[len(attrs)]
            return vec_class(*(self['xy'.index(c)] for c in attrs))
        except Exception:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attrs}'") from None

    def __repr__(self) -> str:
        return f'Vec2({self.x}, {self.y})'