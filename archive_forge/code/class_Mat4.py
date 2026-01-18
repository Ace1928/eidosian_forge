from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
class Mat4(tuple):
    """A 4x4 Matrix

    `Mat4` is an immutable 4x4 Matrix, which includs most common
    operators. This includes class methods for creating orthogonal
    and perspective projection matrixes, to be used by OpenGL.

    A Matrix can be created with a list or tuple of 16 values.
    If no values are provided, an "identity matrix" will be created
    (1.0 on the main diagonal). Mat4 objects are immutable, so
    all operations return a new Mat4 object.

    .. note:: Matrix multiplication is performed using the "@" operator.
    """

    def __new__(cls: type[Mat4T], values: _Iterable[float]=(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)) -> Mat4T:
        new = super().__new__(cls, values)
        assert len(new) == 16, 'A 4x4 Matrix requires 16 values'
        return new

    @classmethod
    def orthogonal_projection(cls: type[Mat4T], left: float, right: float, bottom: float, top: float, z_near: float, z_far: float) -> Mat4T:
        """Create a Mat4 orthographic projection matrix for use with OpenGL.

        Given left, right, bottom, top values, and near/far z planes,
        create a 4x4 Projection Matrix. This is useful for setting
        :py:attr:`~pyglet.window.Window.projection`.
        """
        width = right - left
        height = top - bottom
        depth = z_far - z_near
        sx = 2.0 / width
        sy = 2.0 / height
        sz = 2.0 / -depth
        tx = -(right + left) / width
        ty = -(top + bottom) / height
        tz = -(z_far + z_near) / depth
        return cls((sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, sz, 0.0, tx, ty, tz, 1.0))

    @classmethod
    def perspective_projection(cls: type[Mat4T], aspect: float, z_near: float, z_far: float, fov: float=60) -> Mat4T:
        """Create a Mat4 perspective projection matrix for use with OpenGL.

        Given a desired aspect ratio, near/far planes, and fov (field of view),
        create a 4x4 Projection Matrix. This is useful for setting
        :py:attr:`~pyglet.window.Window.projection`.
        """
        xy_max = z_near * _math.tan(fov * _math.pi / 360)
        y_min = -xy_max
        x_min = -xy_max
        width = xy_max - x_min
        height = xy_max - y_min
        depth = z_far - z_near
        q = -(z_far + z_near) / depth
        qn = -2 * z_far * z_near / depth
        w = 2 * z_near / width
        w = w / aspect
        h = 2 * z_near / height
        return cls((w, 0, 0, 0, 0, h, 0, 0, 0, 0, q, -1, 0, 0, qn, 0))

    @classmethod
    def from_rotation(cls, angle: float, vector: Vec3) -> Mat4:
        """Create a rotation matrix from an angle and Vec3."""
        return cls().rotate(angle, vector)

    @classmethod
    def from_scale(cls: type[Mat4T], vector: Vec3) -> Mat4T:
        """Create a scale matrix from a Vec3."""
        return cls((vector[0], 0.0, 0.0, 0.0, 0.0, vector[1], 0.0, 0.0, 0.0, 0.0, vector[2], 0.0, 0.0, 0.0, 0.0, 1.0))

    @classmethod
    def from_translation(cls: type[Mat4T], vector: Vec3) -> Mat4T:
        """Create a translation matrix from a Vec3."""
        return cls((1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, vector[0], vector[1], vector[2], 1.0))

    @classmethod
    def look_at(cls: type[Mat4T], position: Vec3, target: Vec3, up: Vec3):
        f = (target - position).normalize()
        u = up.normalize()
        s = f.cross(u).normalize()
        u = s.cross(f)
        return cls([s.x, u.x, -f.x, 0.0, s.y, u.y, -f.y, 0.0, s.z, u.z, -f.z, 0.0, -s.dot(position), -u.dot(position), f.dot(position), 1.0])

    def row(self, index: int) -> tuple:
        """Get a specific row as a tuple."""
        return self[index::4]

    def column(self, index: int) -> tuple:
        """Get a specific column as a tuple."""
        return self[index * 4:index * 4 + 4]

    def rotate(self, angle: float, vector: Vec3) -> Mat4:
        """Get a rotation Matrix on x, y, or z axis."""
        if not all((abs(n) <= 1 for n in vector)):
            raise ValueError('vector must be normalized (<=1)')
        x, y, z = vector
        c = _math.cos(angle)
        s = _math.sin(angle)
        t = 1 - c
        temp_x, temp_y, temp_z = (t * x, t * y, t * z)
        ra = c + temp_x * x
        rb = 0 + temp_x * y + s * z
        rc = 0 + temp_x * z - s * y
        re = 0 + temp_y * x - s * z
        rf = c + temp_y * y
        rg = 0 + temp_y * z + s * x
        ri = 0 + temp_z * x + s * y
        rj = 0 + temp_z * y - s * x
        rk = c + temp_z * z
        return Mat4(self) @ Mat4((ra, rb, rc, 0, re, rf, rg, 0, ri, rj, rk, 0, 0, 0, 0, 1))

    def scale(self, vector: Vec3) -> Mat4:
        """Get a scale Matrix on x, y, or z axis."""
        temp = list(self)
        temp[0] *= vector[0]
        temp[5] *= vector[1]
        temp[10] *= vector[2]
        return Mat4(temp)

    def translate(self, vector: Vec3) -> Mat4:
        """Get a translation Matrix along x, y, and z axis."""
        return self @ Mat4((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, *vector, 1))

    def transpose(self) -> Mat4:
        """Get a transpose of this Matrix."""
        return Mat4(self[0::4] + self[1::4] + self[2::4] + self[3::4])

    def __add__(self, other: Mat4) -> Mat4:
        if not isinstance(other, Mat4):
            raise TypeError('Can only add to other Mat4 types')
        return Mat4((s + o for s, o in zip(self, other)))

    def __sub__(self, other: Mat4) -> Mat4:
        if not isinstance(other, Mat4):
            raise TypeError('Can only subtract from other Mat4 types')
        return Mat4((s - o for s, o in zip(self, other)))

    def __pos__(self) -> Mat4:
        return self

    def __neg__(self) -> Mat4:
        return Mat4((-v for v in self))

    def __invert__(self) -> Mat4:
        a = self[10] * self[15] - self[11] * self[14]
        b = self[9] * self[15] - self[11] * self[13]
        c = self[9] * self[14] - self[10] * self[13]
        d = self[8] * self[15] - self[11] * self[12]
        e = self[8] * self[14] - self[10] * self[12]
        f = self[8] * self[13] - self[9] * self[12]
        g = self[6] * self[15] - self[7] * self[14]
        h = self[5] * self[15] - self[7] * self[13]
        i = self[5] * self[14] - self[6] * self[13]
        j = self[6] * self[11] - self[7] * self[10]
        k = self[5] * self[11] - self[7] * self[9]
        l = self[5] * self[10] - self[6] * self[9]
        m = self[4] * self[15] - self[7] * self[12]
        n = self[4] * self[14] - self[6] * self[12]
        o = self[4] * self[11] - self[7] * self[8]
        p = self[4] * self[10] - self[6] * self[8]
        q = self[4] * self[13] - self[5] * self[12]
        r = self[4] * self[9] - self[5] * self[8]
        det = self[0] * (self[5] * a - self[6] * b + self[7] * c) - self[1] * (self[4] * a - self[6] * d + self[7] * e) + self[2] * (self[4] * b - self[5] * d + self[7] * f) - self[3] * (self[4] * c - self[5] * e + self[6] * f)
        if det == 0:
            _warnings.warn('Unable to calculate inverse of singular Matrix')
            return self
        pdet = 1 / det
        ndet = -pdet
        return Mat4((pdet * (self[5] * a - self[6] * b + self[7] * c), ndet * (self[1] * a - self[2] * b + self[3] * c), pdet * (self[1] * g - self[2] * h + self[3] * i), ndet * (self[1] * j - self[2] * k + self[3] * l), ndet * (self[4] * a - self[6] * d + self[7] * e), pdet * (self[0] * a - self[2] * d + self[3] * e), ndet * (self[0] * g - self[2] * m + self[3] * n), pdet * (self[0] * j - self[2] * o + self[3] * p), pdet * (self[4] * b - self[5] * d + self[7] * f), ndet * (self[0] * b - self[1] * d + self[3] * f), pdet * (self[0] * h - self[1] * m + self[3] * q), ndet * (self[0] * k - self[1] * o + self[3] * r), ndet * (self[4] * c - self[5] * e + self[6] * f), pdet * (self[0] * c - self[1] * e + self[2] * f), ndet * (self[0] * i - self[1] * n + self[2] * q), pdet * (self[0] * l - self[1] * p + self[2] * r)))

    def __round__(self, ndigits: int | None=None) -> Mat4:
        return Mat4((round(v, ndigits) for v in self))

    def __mul__(self, other: int) -> _typing.NoReturn:
        raise NotImplementedError('Please use the @ operator for Matrix multiplication.')

    @_typing.overload
    def __matmul__(self, other: Vec4) -> Vec4:
        ...

    @_typing.overload
    def __matmul__(self, other: Mat4) -> Mat4:
        ...

    def __matmul__(self, other):
        if isinstance(other, Vec4):
            r0 = self[0::4]
            r1 = self[1::4]
            r2 = self[2::4]
            r3 = self[3::4]
            return Vec4(sum(map(_mul, r0, other)), sum(map(_mul, r1, other)), sum(map(_mul, r2, other)), sum(map(_mul, r3, other)))
        if not isinstance(other, Mat4):
            raise TypeError('Can only multiply with Mat4 or Vec4 types')
        r0 = self[0::4]
        r1 = self[1::4]
        r2 = self[2::4]
        r3 = self[3::4]
        c0 = other[0:4]
        c1 = other[4:8]
        c2 = other[8:12]
        c3 = other[12:16]
        return Mat4((sum(map(_mul, c0, r0)), sum(map(_mul, c0, r1)), sum(map(_mul, c0, r2)), sum(map(_mul, c0, r3)), sum(map(_mul, c1, r0)), sum(map(_mul, c1, r1)), sum(map(_mul, c1, r2)), sum(map(_mul, c1, r3)), sum(map(_mul, c2, r0)), sum(map(_mul, c2, r1)), sum(map(_mul, c2, r2)), sum(map(_mul, c2, r3)), sum(map(_mul, c3, r0)), sum(map(_mul, c3, r1)), sum(map(_mul, c3, r2)), sum(map(_mul, c3, r3))))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self[0:4]}\n    {self[4:8]}\n    {self[8:12]}\n    {self[12:16]}'