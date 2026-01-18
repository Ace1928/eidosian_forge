from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
def _check_secret_key(secret_key):
    return len(set(secret_key)) >= SECRET_KEY_MIN_UNIQUE_CHARACTERS and len(secret_key) >= SECRET_KEY_MIN_LENGTH and (not secret_key.startswith(SECRET_KEY_INSECURE_PREFIX))