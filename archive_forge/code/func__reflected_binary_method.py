from numpy.core import umath as um
def _reflected_binary_method(ufunc, name):
    """Implement a reflected binary method with a ufunc, e.g., __radd__."""

    def func(self, other):
        if _disables_array_ufunc(other):
            return NotImplemented
        return ufunc(other, self)
    func.__name__ = '__r{}__'.format(name)
    return func