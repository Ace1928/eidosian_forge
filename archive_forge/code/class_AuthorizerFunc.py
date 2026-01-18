import abc
from ._identity import ACLIdentity
class AuthorizerFunc(Authorizer):
    """ Implements a simplified version of Authorizer that operates on a single
    operation at a time.
    """

    def __init__(self, f):
        """
        :param f: a function that takes an identity that operates on a single
        operation at a time. Will return if this op is allowed as a boolean and
        and a list of caveat that holds any additional third party caveats
        that apply.
        """
        self._f = f

    def authorize(self, ctx, identity, ops):
        """Implements Authorizer.authorize by calling f with the given identity
        for each operation.
        """
        allowed = []
        caveats = []
        for op in ops:
            ok, fcaveats = self._f(ctx, identity, op)
            allowed.append(ok)
            if fcaveats is not None:
                caveats.extend(fcaveats)
        return (allowed, caveats)