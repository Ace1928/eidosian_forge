import abc
from ._identity import ACLIdentity
class ClosedAuthorizer(Authorizer):
    """ An Authorizer implementation that will never authorize anything.
    """

    def authorize(self, ctx, id, ops):
        return ([False] * len(ops), [])