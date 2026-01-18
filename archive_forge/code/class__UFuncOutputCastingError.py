from .._utils import set_module
@_display_as_base
class _UFuncOutputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc output cannot be casted """

    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.out_i = i

    def __str__(self):
        i_str = '{} '.format(self.out_i) if self.ufunc.nout != 1 else ''
        return 'Cannot cast ufunc {!r} output {}from {!r} to {!r} with casting rule {!r}'.format(self.ufunc.__name__, i_str, self.from_, self.to, self.casting)