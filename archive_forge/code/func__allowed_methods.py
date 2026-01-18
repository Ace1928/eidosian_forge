import logging
from asgiref.sync import iscoroutinefunction, markcoroutinefunction
from django.core.exceptions import ImproperlyConfigured
from django.http import (
from django.template.response import TemplateResponse
from django.urls import reverse
from django.utils.decorators import classonlymethod
from django.utils.functional import classproperty
def _allowed_methods(self):
    return [m.upper() for m in self.http_method_names if hasattr(self, m)]