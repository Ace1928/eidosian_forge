from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from .. import Error, Tags, Warning, register
def _security_middleware():
    return 'django.middleware.security.SecurityMiddleware' in settings.MIDDLEWARE