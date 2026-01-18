import functools
import warnings
from pathlib import Path
from django.conf import settings
from django.template.backends.django import DjangoTemplates
from django.template.loader import get_template
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
class DjangoDivFormRenderer(DjangoTemplates):
    """
    Load Django templates from django/forms/templates and from apps'
    'templates' directory and use the 'div.html' template to render forms and
    formsets.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn('The DjangoDivFormRenderer transitional form renderer is deprecated. Use DjangoTemplates instead.', RemovedInDjango60Warning)
        super().__init__(*args, **kwargs)