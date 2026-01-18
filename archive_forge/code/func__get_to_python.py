from itertools import chain
from django.core.exceptions import (
from django.db.models.utils import AltersData
from django.forms.fields import ChoiceField, Field
from django.forms.forms import BaseForm, DeclarativeFieldsMetaclass
from django.forms.formsets import BaseFormSet, formset_factory
from django.forms.utils import ErrorList
from django.forms.widgets import (
from django.utils.choices import BaseChoiceIterator
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
def _get_to_python(self, field):
    """
        If the field is a related field, fetch the concrete field's (that
        is, the ultimate pointed-to field's) to_python.
        """
    while field.remote_field is not None:
        field = field.remote_field.get_related_field()
    return field.to_python