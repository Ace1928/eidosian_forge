import string
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.db import models
from django.db.models.signals import pre_delete, pre_save
from django.http.request import split_domain_port
from django.utils.translation import gettext_lazy as _
def _get_site_by_request(self, request):
    host = request.get_host()
    try:
        if host not in SITE_CACHE:
            SITE_CACHE[host] = self.get(domain__iexact=host)
        return SITE_CACHE[host]
    except Site.DoesNotExist:
        domain, port = split_domain_port(host)
        if domain not in SITE_CACHE:
            SITE_CACHE[domain] = self.get(domain__iexact=domain)
        return SITE_CACHE[domain]