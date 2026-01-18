import inspect
from importlib import import_module
from inspect import cleandoc
from pathlib import Path
from django.apps import apps
from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.admindocs import utils
from django.contrib.admindocs.utils import (
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.db import models
from django.http import Http404
from django.template.engine import Engine
from django.urls import get_mod_func, get_resolver, get_urlconf
from django.utils._os import safe_join
from django.utils.decorators import method_decorator
from django.utils.functional import cached_property
from django.utils.inspect import (
from django.utils.translation import gettext as _
from django.views.generic import TemplateView
from .utils import get_view_name
class TemplateFilterIndexView(BaseAdminDocsView):
    template_name = 'admin_doc/template_filter_index.html'

    def get_context_data(self, **kwargs):
        filters = []
        try:
            engine = Engine.get_default()
        except ImproperlyConfigured:
            pass
        else:
            app_libs = sorted(engine.template_libraries.items())
            builtin_libs = [('', lib) for lib in engine.template_builtins]
            for module_name, library in builtin_libs + app_libs:
                for filter_name, filter_func in library.filters.items():
                    title, body, metadata = utils.parse_docstring(filter_func.__doc__)
                    title = title and utils.parse_rst(title, 'filter', _('filter:') + filter_name)
                    body = body and utils.parse_rst(body, 'filter', _('filter:') + filter_name)
                    for key in metadata:
                        metadata[key] = utils.parse_rst(metadata[key], 'filter', _('filter:') + filter_name)
                    tag_library = module_name.split('.')[-1]
                    filters.append({'name': filter_name, 'title': title, 'body': body, 'meta': metadata, 'library': tag_library})
        return super().get_context_data(**{**kwargs, 'filters': filters})