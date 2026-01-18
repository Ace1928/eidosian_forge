import inspect
from django.conf import settings
from .. import Error, Tags, Warning, register
def _csrf_middleware():
    return 'django.middleware.csrf.CsrfViewMiddleware' in settings.MIDDLEWARE