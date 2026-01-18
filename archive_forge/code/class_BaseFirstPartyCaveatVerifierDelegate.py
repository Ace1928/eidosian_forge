from abc import ABCMeta, abstractmethod
class BaseFirstPartyCaveatVerifierDelegate(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(BaseFirstPartyCaveatVerifierDelegate, self).__init__(*args, **kwargs)

    @abstractmethod
    def verify_first_party_caveat(self, verifier, caveat, signature):
        pass

    @abstractmethod
    def update_signature(self, signature, caveat):
        pass