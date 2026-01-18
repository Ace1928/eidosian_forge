from asgiref.local import Local
from django.conf import settings as django_settings
from django.utils.functional import cached_property
class ConnectionDoesNotExist(Exception):
    pass