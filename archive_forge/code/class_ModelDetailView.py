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
class ModelDetailView(BaseAdminDocsView):
    template_name = 'admin_doc/model_detail.html'

    def get_context_data(self, **kwargs):
        model_name = self.kwargs['model_name']
        try:
            app_config = apps.get_app_config(self.kwargs['app_label'])
        except LookupError:
            raise Http404(_('App %(app_label)r not found') % self.kwargs)
        try:
            model = app_config.get_model(model_name)
        except LookupError:
            raise Http404(_('Model %(model_name)r not found in app %(app_label)r') % self.kwargs)
        opts = model._meta
        title, body, metadata = utils.parse_docstring(model.__doc__)
        title = title and utils.parse_rst(title, 'model', _('model:') + model_name)
        body = body and utils.parse_rst(body, 'model', _('model:') + model_name)
        fields = []
        for field in opts.fields:
            if isinstance(field, models.ForeignKey):
                data_type = field.remote_field.model.__name__
                app_label = field.remote_field.model._meta.app_label
                verbose = utils.parse_rst(_('the related `%(app_label)s.%(data_type)s` object') % {'app_label': app_label, 'data_type': data_type}, 'model', _('model:') + data_type)
            else:
                data_type = get_readable_field_data_type(field)
                verbose = field.verbose_name
            fields.append({'name': field.name, 'data_type': data_type, 'verbose': verbose or '', 'help_text': field.help_text})
        for field in opts.many_to_many:
            data_type = field.remote_field.model.__name__
            app_label = field.remote_field.model._meta.app_label
            verbose = _('related `%(app_label)s.%(object_name)s` objects') % {'app_label': app_label, 'object_name': data_type}
            fields.append({'name': '%s.all' % field.name, 'data_type': 'List', 'verbose': utils.parse_rst(_('all %s') % verbose, 'model', _('model:') + opts.model_name)})
            fields.append({'name': '%s.count' % field.name, 'data_type': 'Integer', 'verbose': utils.parse_rst(_('number of %s') % verbose, 'model', _('model:') + opts.model_name)})
        methods = []
        for func_name, func in model.__dict__.items():
            if inspect.isfunction(func) or isinstance(func, (cached_property, property)):
                try:
                    for exclude in MODEL_METHODS_EXCLUDE:
                        if func_name.startswith(exclude):
                            raise StopIteration
                except StopIteration:
                    continue
                verbose = func.__doc__
                verbose = verbose and utils.parse_rst(cleandoc(verbose), 'model', _('model:') + opts.model_name)
                if isinstance(func, (cached_property, property)):
                    fields.append({'name': func_name, 'data_type': get_return_data_type(func_name), 'verbose': verbose or ''})
                elif method_has_no_args(func) and (not func_accepts_kwargs(func)) and (not func_accepts_var_args(func)):
                    fields.append({'name': func_name, 'data_type': get_return_data_type(func_name), 'verbose': verbose or ''})
                else:
                    arguments = get_func_full_args(func)
                    print_arguments = ', '.join(['='.join([arg_el[0], *map(repr, arg_el[1:])]) for arg_el in arguments])
                    methods.append({'name': func_name, 'arguments': print_arguments, 'verbose': verbose or ''})
        for rel in opts.related_objects:
            verbose = _('related `%(app_label)s.%(object_name)s` objects') % {'app_label': rel.related_model._meta.app_label, 'object_name': rel.related_model._meta.object_name}
            accessor = rel.get_accessor_name()
            fields.append({'name': '%s.all' % accessor, 'data_type': 'List', 'verbose': utils.parse_rst(_('all %s') % verbose, 'model', _('model:') + opts.model_name)})
            fields.append({'name': '%s.count' % accessor, 'data_type': 'Integer', 'verbose': utils.parse_rst(_('number of %s') % verbose, 'model', _('model:') + opts.model_name)})
        return super().get_context_data(**{**kwargs, 'name': opts.label, 'summary': title, 'description': body, 'fields': fields, 'methods': methods})