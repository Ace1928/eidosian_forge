from .._utils import set_module
@_display_as_base
class _UFuncBinaryResolutionError(_UFuncNoLoopError):
    """ Thrown when a binary resolution fails """

    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc, dtypes)
        assert len(self.dtypes) == 2

    def __str__(self):
        return 'ufunc {!r} cannot use operands with types {!r} and {!r}'.format(self.ufunc.__name__, *self.dtypes)