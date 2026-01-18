import inspect
import os
from importlib import import_module
from django.core.exceptions import ImproperlyConfigured
from django.utils.functional import cached_property
from django.utils.module_loading import import_string, module_has_submodule
@property
def _is_default_auto_field_overridden(self):
    return self.__class__.default_auto_field is not AppConfig.default_auto_field