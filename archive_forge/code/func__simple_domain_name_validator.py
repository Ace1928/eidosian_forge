import string
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.db.models.signals import pre_delete, pre_save
from django.http.request import split_domain_port
from django.utils.translation import gettext_lazy as _
def _simple_domain_name_validator(value):
    """
    Validate that the given value contains no whitespaces to prevent common
    typos.
    """
    checks = (s in value for s in string.whitespace)
    if any(checks):
        raise ValidationError(_('The domain name cannot contain any spaces or tabs.'), code='invalid')