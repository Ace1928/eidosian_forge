from django.conf import settings
from .. import Tags, Warning, register
def _session_middleware():
    return 'django.contrib.sessions.middleware.SessionMiddleware' in settings.MIDDLEWARE