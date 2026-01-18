from django.core.exceptions import ImproperlyConfigured
from django.forms import Form
from django.forms import models as model_forms
from django.http import HttpResponseRedirect
from django.views.generic.base import ContextMixin, TemplateResponseMixin, View
from django.views.generic.detail import (
class BaseFormView(FormMixin, ProcessFormView):
    """A base view for displaying a form."""