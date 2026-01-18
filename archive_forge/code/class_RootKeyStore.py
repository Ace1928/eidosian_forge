import abc
import os
class RootKeyStore(object):
    """ Defines a store for macaroon root keys.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get(self, id):
        """ Returns the root key for the given id.
        If the item is not there, it returns None.
        @param id: bytes
        @return: bytes
        """
        raise NotImplementedError('get method must be defined in subclass')

    @abc.abstractmethod
    def root_key(self):
        """ Returns the root key to be used for making a new macaroon, and an
        id that can be used to look it up later with the get method.
        Note that the root keys should remain available for as long as the
        macaroons using them are valid.
        Note that there is no need for it to return a new root key for every
        call - keys may be reused, although some key cycling is over time is
        advisable.
        @return: bytes
        """