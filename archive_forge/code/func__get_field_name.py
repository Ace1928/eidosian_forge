from django.conf import settings
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models
def _get_field_name(self):
    """Return self.__field_name or 'site' or 'sites'."""
    if not self.__field_name:
        try:
            self.model._meta.get_field('site')
        except FieldDoesNotExist:
            self.__field_name = 'sites'
        else:
            self.__field_name = 'site'
    return self.__field_name