from django.conf import settings
from .. import Tags, Warning, register
def _session_app():
    return 'django.contrib.sessions' in settings.INSTALLED_APPS