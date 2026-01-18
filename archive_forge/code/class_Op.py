from collections import namedtuple
from threading import Lock
from ._authorizer import ClosedAuthorizer
from ._identity import NoIdentities
from ._error import (
import macaroonbakery.checkers as checkers
import pyrfc3339
class Op(namedtuple('Op', 'entity, action')):
    """ Op holds an entity and action to be authorized on that entity.
    entity string holds the name of the entity to be authorized.

    @param entity should not contain spaces and should
    not start with the prefix "login" or "multi-" (conventionally,
    entity names will be prefixed with the entity type followed
    by a hyphen.
    @param action string holds the action to perform on the entity,
    such as "read" or "delete". It is up to the service using a checker
    to define a set of operations and keep them consistent over time.
    """