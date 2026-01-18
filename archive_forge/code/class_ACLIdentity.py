import abc
from ._error import IdentityError
class ACLIdentity(Identity):
    """ ACLIdentity may be implemented by Identity implementations
    to report group membership information.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def allow(self, ctx, acls):
        """ reports whether the user should be allowed to access
        any of the users or groups in the given acl list.
        :param ctx(AuthContext) is the context of the authorization request.
        :param acls array of string acl
        :return boolean
        """
        raise NotImplementedError('allow method must be defined in subclass')