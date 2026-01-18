import abc
from ._identity import ACLIdentity
Implements Authorizer.authorize by calling identity.allow to
        determine whether the identity is a member of the ACLs associated with
        the given operations.
        