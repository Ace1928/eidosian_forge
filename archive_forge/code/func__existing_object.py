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
def _existing_object(self, pk):
    if not hasattr(self, '_object_dict'):
        self._object_dict = {o.pk: o for o in self.get_queryset()}
    return self._object_dict.get(pk)