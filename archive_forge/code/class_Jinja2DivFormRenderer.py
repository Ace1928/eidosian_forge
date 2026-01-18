import functools
import warnings
from pathlib import Path
from django.conf import settings
from django.template.backends.django import DjangoTemplates
from django.template.loader import get_template
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
class Jinja2DivFormRenderer(Jinja2):
    """
    Load Jinja2 templates from the built-in widget templates in
    django/forms/jinja2 and from apps' 'jinja2' directory.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn('The Jinja2DivFormRenderer transitional form renderer is deprecated. Use Jinja2 instead.', RemovedInDjango60Warning)
        super().__init__(*args, **kwargs)