from abc import ABCMeta, abstractmethod
class BaseFirstPartyCaveatDelegate(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(BaseFirstPartyCaveatDelegate, self).__init__(*args, **kwargs)

    @abstractmethod
    def add_first_party_caveat(self, macaroon, predicate, **kwargs):
        pass