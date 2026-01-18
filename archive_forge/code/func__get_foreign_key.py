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
def _get_foreign_key(parent_model, model, fk_name=None, can_fail=False):
    """
    Find and return the ForeignKey from model to parent if there is one
    (return None if can_fail is True and no such field exists). If fk_name is
    provided, assume it is the name of the ForeignKey field. Unless can_fail is
    True, raise an exception if there isn't a ForeignKey from model to
    parent_model.
    """
    from django.db.models import ForeignKey
    opts = model._meta
    if fk_name:
        fks_to_parent = [f for f in opts.fields if f.name == fk_name]
        if len(fks_to_parent) == 1:
            fk = fks_to_parent[0]
            parent_list = parent_model._meta.get_parent_list()
            if not isinstance(fk, ForeignKey) or (fk.remote_field.model._meta.proxy and fk.remote_field.model._meta.proxy_for_model not in parent_list) or (not fk.remote_field.model._meta.proxy and fk.remote_field.model != parent_model and (fk.remote_field.model not in parent_list)):
                raise ValueError("fk_name '%s' is not a ForeignKey to '%s'." % (fk_name, parent_model._meta.label))
        elif not fks_to_parent:
            raise ValueError("'%s' has no field named '%s'." % (model._meta.label, fk_name))
    else:
        parent_list = parent_model._meta.get_parent_list()
        fks_to_parent = [f for f in opts.fields if isinstance(f, ForeignKey) and (f.remote_field.model == parent_model or f.remote_field.model in parent_list or (f.remote_field.model._meta.proxy and f.remote_field.model._meta.proxy_for_model in parent_list))]
        if len(fks_to_parent) == 1:
            fk = fks_to_parent[0]
        elif not fks_to_parent:
            if can_fail:
                return
            raise ValueError("'%s' has no ForeignKey to '%s'." % (model._meta.label, parent_model._meta.label))
        else:
            raise ValueError("'%s' has more than one ForeignKey to '%s'. You must specify a 'fk_name' attribute." % (model._meta.label, parent_model._meta.label))
    return fk