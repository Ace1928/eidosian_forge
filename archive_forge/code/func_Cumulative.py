from sys import version_info as _swig_python_version_info
import weakref
def Cumulative(self, *args):
    """
        *Overload 1:*
        This constraint forces that, for any integer t, the sum of the demands
        corresponding to an interval containing t does not exceed the given
        capacity.

        Intervals and demands should be vectors of equal size.

        Demands should only contain non-negative values. Zero values are
        supported, and the corresponding intervals are filtered out, as they
        neither impact nor are impacted by this constraint.

        |

        *Overload 2:*
        This constraint forces that, for any integer t, the sum of the demands
        corresponding to an interval containing t does not exceed the given
        capacity.

        Intervals and demands should be vectors of equal size.

        Demands should only contain non-negative values. Zero values are
        supported, and the corresponding intervals are filtered out, as they
        neither impact nor are impacted by this constraint.

        |

        *Overload 3:*
        This constraint forces that, for any integer t, the sum of the demands
        corresponding to an interval containing t does not exceed the given
        capacity.

        Intervals and demands should be vectors of equal size.

        Demands should only contain non-negative values. Zero values are
        supported, and the corresponding intervals are filtered out, as they
        neither impact nor are impacted by this constraint.

        |

        *Overload 4:*
        This constraint enforces that, for any integer t, the sum of the demands
        corresponding to an interval containing t does not exceed the given
        capacity.

        Intervals and demands should be vectors of equal size.

        Demands should only contain non-negative values. Zero values are
        supported, and the corresponding intervals are filtered out, as they
        neither impact nor are impacted by this constraint.

        |

        *Overload 5:*
        This constraint enforces that, for any integer t, the sum of demands
        corresponding to an interval containing t does not exceed the given
        capacity.

        Intervals and demands should be vectors of equal size.

        Demands should be positive.

        |

        *Overload 6:*
        This constraint enforces that, for any integer t, the sum of demands
        corresponding to an interval containing t does not exceed the given
        capacity.

        Intervals and demands should be vectors of equal size.

        Demands should be positive.
        """
    return _pywrapcp.Solver_Cumulative(self, *args)