from .utils import typename
class VariadicSignatureMeta(type):
    """A metaclass that overrides ``__getitem__`` on the class. This is used to
    generate a new type for Variadic signatures. See the Variadic class for
    examples of how this behaves.
    """

    def __getitem__(cls, variadic_type):
        if not (isinstance(variadic_type, (type, tuple)) or type(variadic_type)):
            raise ValueError('Variadic types must be type or tuple of types (Variadic[int] or Variadic[(int, float)]')
        if not isinstance(variadic_type, tuple):
            variadic_type = (variadic_type,)
        return VariadicSignatureType(f'Variadic[{typename(variadic_type)}]', (), dict(variadic_type=variadic_type, __slots__=()))