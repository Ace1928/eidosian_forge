import hashlib
import hmac
import secrets
from django.conf import settings
from django.utils.encoding import force_bytes
class InvalidAlgorithm(ValueError):
    """Algorithm is not supported by hashlib."""
    pass