from django import forms
from django.core import exceptions
from django.db.backends.postgresql.psycopg_any import (
from django.forms.widgets import HiddenInput, MultiWidget
from django.utils.translation import gettext_lazy as _
class HiddenRangeWidget(RangeWidget):
    """A widget that splits input into two <input type="hidden"> inputs."""

    def __init__(self, attrs=None):
        super().__init__(HiddenInput, attrs)