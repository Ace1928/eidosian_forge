from __future__ import unicode_literals
class LayoutDimension(object):
    """
    Specified dimension (width/height) of a user control or window.

    The layout engine tries to honor the preferred size. If that is not
    possible, because the terminal is larger or smaller, it tries to keep in
    between min and max.

    :param min: Minimum size.
    :param max: Maximum size.
    :param weight: For a VSplit/HSplit, the actual size will be determined
                   by taking the proportion of weights from all the children.
                   E.g. When there are two children, one width a weight of 1,
                   and the other with a weight of 2. The second will always be
                   twice as big as the first, if the min/max values allow it.
    :param preferred: Preferred size.
    """

    def __init__(self, min=None, max=None, weight=1, preferred=None):
        assert isinstance(weight, int) and weight > 0
        self.min_specified = min is not None
        self.max_specified = max is not None
        self.preferred_specified = preferred is not None
        if min is None:
            min = 0
        if max is None:
            max = 1000 ** 10
        if preferred is None:
            preferred = min
        self.min = min
        self.max = max
        self.preferred = preferred
        self.weight = weight
        if self.preferred < self.min:
            self.preferred = self.min
        if self.preferred > self.max:
            self.preferred = self.max

    @classmethod
    def exact(cls, amount):
        """
        Return a :class:`.LayoutDimension` with an exact size. (min, max and
        preferred set to ``amount``).
        """
        return cls(min=amount, max=amount, preferred=amount)

    def __repr__(self):
        return 'LayoutDimension(min=%r, max=%r, preferred=%r, weight=%r)' % (self.min, self.max, self.preferred, self.weight)

    def __add__(self, other):
        return sum_layout_dimensions([self, other])