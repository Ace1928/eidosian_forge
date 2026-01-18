from math import sqrt
def _errt(self):
    """Count Error-Rate Relative to Truncation (ERRT).

        :return: ERRT, length of the line from origo to (UI, OI) divided by
        the length of the line from origo to the point defined by the same
        line when extended until the truncation line.
        :rtype: float
        """
    self.coords = self._get_truncation_coordinates()
    if (0.0, 0.0) in self.coords:
        if (self.ui, self.oi) != (0.0, 0.0):
            return float('inf')
        else:
            return float('nan')
    if (self.ui, self.oi) == (0.0, 0.0):
        return 0.0
    intersection = _count_intersection(((0, 0), (self.ui, self.oi)), self.coords[-2:])
    op = sqrt(self.ui ** 2 + self.oi ** 2)
    ot = sqrt(intersection[0] ** 2 + intersection[1] ** 2)
    return op / ot