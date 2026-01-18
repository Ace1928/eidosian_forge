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
def _save_m2m(self):
    """
        Save the many-to-many fields and generic relations for this form.
        """
    cleaned_data = self.cleaned_data
    exclude = self._meta.exclude
    fields = self._meta.fields
    opts = self.instance._meta
    for f in chain(opts.many_to_many, opts.private_fields):
        if not hasattr(f, 'save_form_data'):
            continue
        if fields and f.name not in fields:
            continue
        if exclude and f.name in exclude:
            continue
        if f.name in cleaned_data:
            f.save_form_data(self.instance, cleaned_data[f.name])