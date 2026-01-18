from django.views.generic.base import RedirectView, TemplateView, View
from django.views.generic.dates import (
from django.views.generic.detail import DetailView
from django.views.generic.edit import CreateView, DeleteView, FormView, UpdateView
from django.views.generic.list import ListView
class GenericViewError(Exception):
    """A problem in a generic view."""
    pass