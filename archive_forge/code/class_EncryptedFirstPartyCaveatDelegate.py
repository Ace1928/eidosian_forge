from __future__ import unicode_literals
import binascii
from six import iteritems
from pymacaroons.field_encryptors import SecretBoxEncryptor
from .first_party import (
class EncryptedFirstPartyCaveatDelegate(FirstPartyCaveatDelegate):

    def __init__(self, field_encryptor=None, *args, **kwargs):
        self.field_encryptor = field_encryptor or SecretBoxEncryptor()
        super(EncryptedFirstPartyCaveatDelegate, self).__init__(*args, **kwargs)

    def add_first_party_caveat(self, macaroon, predicate, **kwargs):
        if kwargs.get('encrypted'):
            predicate = self.field_encryptor.encrypt(binascii.unhexlify(macaroon.signature_bytes), predicate)
        return super(EncryptedFirstPartyCaveatDelegate, self).add_first_party_caveat(macaroon, predicate, **kwargs)