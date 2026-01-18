import binascii
import copy
import random
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import private_key as pri_key
from castellan.common.objects import public_key as pub_key
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import key_manager
def add_consumer(self, context, managed_object_id, consumer_data):
    if context is None:
        raise exception.Forbidden()
    if managed_object_id not in self.keys:
        raise exception.ManagedObjectNotFoundError()
    self.keys[managed_object_id].consumers.append(consumer_data)