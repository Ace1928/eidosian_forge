from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
class ArrheniusParamWithUnits(ArrheniusParam):

    @classmethod
    def from_rateconst_at_T(cls, *args, **kwargs):
        if 'constants' not in kwargs:
            kwargs['constants'] = default_constants
        if 'units' not in kwargs:
            kwargs['units'] = default_units
        if 'backend' not in kwargs:
            kwargs['backend'] = patched_numpy
        return super(ArrheniusParamWithUnits, cls).from_rateconst_at_T(*args, **kwargs)

    def __call__(self, state, constants=default_constants, units=default_units, backend=None):
        """See :func:`chempy.arrhenius.arrhenius_equation`."""
        return super(ArrheniusParamWithUnits, self).__call__(state, constants, units, backend)

    def as_RateExpr(self, unique_keys=None, constants=default_constants, units=default_units):
        return super(ArrheniusParamWithUnits, self).as_RateExpr(unique_keys, constants, units)