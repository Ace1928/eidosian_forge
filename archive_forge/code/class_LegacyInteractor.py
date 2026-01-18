import abc
from collections import namedtuple
class LegacyInteractor(object):
    """ May optionally be implemented by Interactor implementations that
    implement the legacy interaction-required error protocols.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def legacy_interact(self, client, location, visit_url):
        """ Implements the "visit" half of a legacy discharge
        interaction. The "wait" half will be implemented by httpbakery.
        The location is the location specified by the third party
        caveat. The client holds the client being used to do the current
        request.
        @param client The client being used for the current request {Client}
        @param location Third party caveat location {str}
        @param visit_url The visit_url field from the error {str}
        @return None
        """
        raise NotImplementedError('legacy_interact method must be defined in subclass')