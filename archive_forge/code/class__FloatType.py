import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class _FloatType(_NumericType, sqltypes.Float):

    def __init__(self, precision=None, scale=None, asdecimal=True, **kw):
        if isinstance(self, (REAL, DOUBLE)) and (precision is None and scale is not None or (precision is not None and scale is None)):
            raise exc.ArgumentError('You must specify both precision and scale or omit both altogether.')
        super().__init__(precision=precision, asdecimal=asdecimal, **kw)
        self.scale = scale

    def __repr__(self):
        return util.generic_repr(self, to_inspect=[_FloatType, _NumericType, sqltypes.Float])