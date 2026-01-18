from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
class FranckCondonRecursive:
    """Recursive implementation of Franck-Condon overlaps

    Notes
    -----
    The ovelaps are signed according to the sign of the displacements.

    Reference
    ---------
    Julien Guthmuller
    The Journal of Chemical Physics 144, 064106 (2016); doi: 10.1063/1.4941449
    """

    def __init__(self):
        self.factorial = Factorial()

    def ov0m(self, m, delta):
        if m == 0:
            return np.exp(-0.25 * delta ** 2)
        else:
            assert m > 0
            return -delta / np.sqrt(2 * m) * self.ov0m(m - 1, delta)

    def ov1m(self, m, delta):
        sum = delta * self.ov0m(m, delta) / np.sqrt(2.0)
        if m == 0:
            return sum
        else:
            assert m > 0
            return sum + np.sqrt(m) * self.ov0m(m - 1, delta)

    def ov2m(self, m, delta):
        sum = delta * self.ov1m(m, delta) / 2
        if m == 0:
            return sum
        else:
            assert m > 0
            return sum + np.sqrt(m / 2.0) * self.ov1m(m - 1, delta)

    def ov3m(self, m, delta):
        sum = delta * self.ov2m(m, delta) / np.sqrt(6.0)
        if m == 0:
            return sum
        else:
            assert m > 0
            return sum + np.sqrt(m / 3.0) * self.ov2m(m - 1, delta)

    def ov0mm1(self, m, delta):
        if m == 0:
            return delta / np.sqrt(2) * self.ov0m(m, delta) ** 2
        else:
            return delta / np.sqrt(2) * (self.ov0m(m, delta) ** 2 - self.ov0m(m - 1, delta) ** 2)

    def direct0mm1(self, m, delta):
        """direct and fast <0|m><m|1>"""
        S = delta ** 2 / 2.0
        sum = S ** m
        if m:
            sum -= m * S ** (m - 1)
        return np.where(S == 0, 0, np.exp(-S) * delta / np.sqrt(2) * sum * self.factorial.inv(m))

    def ov0mm2(self, m, delta):
        if m == 0:
            return delta ** 2 / np.sqrt(8) * self.ov0m(m, delta) ** 2
        elif m == 1:
            return delta ** 2 / np.sqrt(8) * (self.ov0m(m, delta) ** 2 - 2 * self.ov0m(m - 1, delta) ** 2)
        else:
            return delta ** 2 / np.sqrt(8) * (self.ov0m(m, delta) ** 2 - 2 * self.ov0m(m - 1, delta) ** 2 + self.ov0m(m - 2, delta) ** 2)

    def direct0mm2(self, m, delta):
        """direct and fast <0|m><m|2>"""
        S = delta ** 2 / 2.0
        sum = S ** (m + 1)
        if m >= 1:
            sum -= 2 * m * S ** m
        if m >= 2:
            sum += m * (m - 1) * S ** (m - 1)
        return np.exp(-S) / np.sqrt(2) * sum * self.factorial.inv(m)

    def ov1mm2(self, m, delta):
        p1 = delta ** 3 / 4.0
        sum = p1 * self.ov0m(m, delta) ** 2
        if m == 0:
            return sum
        p2 = delta - 3.0 * delta ** 3 / 4
        sum += p2 * self.ov0m(m - 1, delta) ** 2
        if m == 1:
            return sum
        sum -= p2 * self.ov0m(m - 2, delta) ** 2
        if m == 2:
            return sum
        return sum - p1 * self.ov0m(m - 3, delta) ** 2

    def direct1mm2(self, m, delta):
        S = delta ** 2 / 2.0
        sum = S ** 2
        if m > 0:
            sum -= 2 * m * S
        if m > 1:
            sum += m * (m - 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(S == 0, 0, np.exp(-S) * S ** (m - 1) / delta * (S - m) * sum * self.factorial.inv(m))

    def direct0mm3(self, m, delta):
        S = delta ** 2 / 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(S == 0, 0, np.exp(-S) * S ** (m - 1) / delta * np.sqrt(12.0) * (S ** 3 / 6.0 - m * S ** 2 / 2 + m * (m - 1) * S / 2.0 - m * (m - 1) * (m - 2) / 6) * self.factorial.inv(m))