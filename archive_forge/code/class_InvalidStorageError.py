from django.conf import DEFAULT_STORAGE_ALIAS, STATICFILES_STORAGE_ALIAS, settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
class InvalidStorageError(ImproperlyConfigured):
    pass