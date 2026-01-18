from .utils import typename
class VariadicSignatureType(type):

    def __subclasscheck__(cls, subclass):
        other_type = subclass.variadic_type if isvariadic(subclass) else (subclass,)
        return subclass is cls or all((issubclass(other, cls.variadic_type) for other in other_type))

    def __eq__(cls, other):
        """
        Return True if other has the same variadic type
        Parameters
        ----------
        other : object (type)
            The object (type) to check
        Returns
        -------
        bool
            Whether or not `other` is equal to `self`
        """
        return isvariadic(other) and set(cls.variadic_type) == set(other.variadic_type)

    def __hash__(cls):
        return hash((type(cls), frozenset(cls.variadic_type)))