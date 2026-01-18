import string
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.db.models.signals import pre_delete, pre_save
from django.http.request import split_domain_port
from django.utils.translation import gettext_lazy as _
def _get_site_by_id(self, site_id):
    if site_id not in SITE_CACHE:
        site = self.get(pk=site_id)
        SITE_CACHE[site_id] = site
    return SITE_CACHE[site_id]