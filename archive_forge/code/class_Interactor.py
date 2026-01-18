import abc
from collections import namedtuple
class Interactor(object):
    """ Represents a way of persuading a discharger that it should grant a
    discharge macaroon.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def kind(self):
        """Returns the interaction method name. This corresponds to the key in
        the Error.interaction_methods type.
        @return {str}
        """
        raise NotImplementedError('kind method must be defined in subclass')

    def interact(self, client, location, interaction_required_err):
        """ Performs the interaction, and returns a token that can be
        used to acquire the discharge macaroon. The location provides
        the third party caveat location to make it possible to use
        relative URLs. The client holds the client being used to do the current
        request.

        If the given interaction isn't supported by the client for
        the given location, it may raise an InteractionMethodNotFound
        which will cause the interactor to be ignored that time.
        @param client The client being used for the current request {Client}
        @param location Third party caveat location {str}
        @param interaction_required_err The error causing the interaction to
        take place {Error}
        @return {DischargeToken} The discharge token.
        """
        raise NotImplementedError('interact method must be defined in subclass')