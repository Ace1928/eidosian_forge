from django.core.exceptions import ImproperlyConfigured
from django.forms import Form
from django.forms import models as model_forms
from django.http import HttpResponseRedirect
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
from django.views.generic.detail import (
class UpdateView(SingleObjectTemplateResponseMixin, BaseUpdateView):
    """View for updating an object, with a response rendered by a template."""
    template_name_suffix = '_form'