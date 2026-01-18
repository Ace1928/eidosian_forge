from abc import ABCMeta, abstractmethod
class BaseThirdPartyCaveatVerifierDelegate(object):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(BaseThirdPartyCaveatVerifierDelegate, self).__init__(*args, **kwargs)

    @abstractmethod
    def verify_third_party_caveat(self, verifier, caveat, root, macaroon, discharge_macaroons, signature):
        pass

    @abstractmethod
    def update_signature(self, signature, caveat):
        pass