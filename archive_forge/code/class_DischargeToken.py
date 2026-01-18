import abc
from collections import namedtuple
class DischargeToken(namedtuple('DischargeToken', 'kind, value')):
    """ Holds a token that is intended to persuade a discharger to discharge
    a third party caveat.
    @param kind holds the kind of the token. By convention this
    matches the name of the interaction method used to
    obtain the token, but that's not required {str}
    @param value holds the token data. {bytes}
    """