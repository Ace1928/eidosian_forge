import collections
from itertools import chain
from django.apps import apps
from django.conf import settings
from django.contrib.admin.exceptions import NotRegistered
from django.contrib.admin.utils import NotRelationField, flatten, get_fields_from_path
from django.core import checks
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import Combinable
from django.forms.models import BaseModelForm, BaseModelFormSet, _get_foreign_key
from django.template import engines
from django.template.backends.django import DjangoTemplates
from django.utils.module_loading import import_string
class BaseModelAdminChecks:

    def check(self, admin_obj, **kwargs):
        return [*self._check_autocomplete_fields(admin_obj), *self._check_raw_id_fields(admin_obj), *self._check_fields(admin_obj), *self._check_fieldsets(admin_obj), *self._check_exclude(admin_obj), *self._check_form(admin_obj), *self._check_filter_vertical(admin_obj), *self._check_filter_horizontal(admin_obj), *self._check_radio_fields(admin_obj), *self._check_prepopulated_fields(admin_obj), *self._check_view_on_site_url(admin_obj), *self._check_ordering(admin_obj), *self._check_readonly_fields(admin_obj)]

    def _check_autocomplete_fields(self, obj):
        """
        Check that `autocomplete_fields` is a list or tuple of model fields.
        """
        if not isinstance(obj.autocomplete_fields, (list, tuple)):
            return must_be('a list or tuple', option='autocomplete_fields', obj=obj, id='admin.E036')
        else:
            return list(chain.from_iterable([self._check_autocomplete_fields_item(obj, field_name, 'autocomplete_fields[%d]' % index) for index, field_name in enumerate(obj.autocomplete_fields)]))

    def _check_autocomplete_fields_item(self, obj, field_name, label):
        """
        Check that an item in `autocomplete_fields` is a ForeignKey or a
        ManyToManyField and that the item has a related ModelAdmin with
        search_fields defined.
        """
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E037')
        else:
            if not field.many_to_many and (not isinstance(field, models.ForeignKey)):
                return must_be('a foreign key or a many-to-many field', option=label, obj=obj, id='admin.E038')
            try:
                related_admin = obj.admin_site.get_model_admin(field.remote_field.model)
            except NotRegistered:
                return [checks.Error('An admin for model "%s" has to be registered to be referenced by %s.autocomplete_fields.' % (field.remote_field.model.__name__, type(obj).__name__), obj=obj.__class__, id='admin.E039')]
            else:
                if not related_admin.search_fields:
                    return [checks.Error('%s must define "search_fields", because it\'s referenced by %s.autocomplete_fields.' % (related_admin.__class__.__name__, type(obj).__name__), obj=obj.__class__, id='admin.E040')]
            return []

    def _check_raw_id_fields(self, obj):
        """Check that `raw_id_fields` only contains field names that are listed
        on the model."""
        if not isinstance(obj.raw_id_fields, (list, tuple)):
            return must_be('a list or tuple', option='raw_id_fields', obj=obj, id='admin.E001')
        else:
            return list(chain.from_iterable((self._check_raw_id_fields_item(obj, field_name, 'raw_id_fields[%d]' % index) for index, field_name in enumerate(obj.raw_id_fields))))

    def _check_raw_id_fields_item(self, obj, field_name, label):
        """Check an item of `raw_id_fields`, i.e. check that field named
        `field_name` exists in model `model` and is a ForeignKey or a
        ManyToManyField."""
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E002')
        else:
            if field.name != field_name:
                return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E002')
            if not field.many_to_many and (not isinstance(field, models.ForeignKey)):
                return must_be('a foreign key or a many-to-many field', option=label, obj=obj, id='admin.E003')
            else:
                return []

    def _check_fields(self, obj):
        """Check that `fields` only refer to existing fields, doesn't contain
        duplicates. Check if at most one of `fields` and `fieldsets` is defined.
        """
        if obj.fields is None:
            return []
        elif not isinstance(obj.fields, (list, tuple)):
            return must_be('a list or tuple', option='fields', obj=obj, id='admin.E004')
        elif obj.fieldsets:
            return [checks.Error("Both 'fieldsets' and 'fields' are specified.", obj=obj.__class__, id='admin.E005')]
        fields = flatten(obj.fields)
        if len(fields) != len(set(fields)):
            return [checks.Error("The value of 'fields' contains duplicate field(s).", obj=obj.__class__, id='admin.E006')]
        return list(chain.from_iterable((self._check_field_spec(obj, field_name, 'fields') for field_name in obj.fields)))

    def _check_fieldsets(self, obj):
        """Check that fieldsets is properly formatted and doesn't contain
        duplicates."""
        if obj.fieldsets is None:
            return []
        elif not isinstance(obj.fieldsets, (list, tuple)):
            return must_be('a list or tuple', option='fieldsets', obj=obj, id='admin.E007')
        else:
            seen_fields = []
            return list(chain.from_iterable((self._check_fieldsets_item(obj, fieldset, 'fieldsets[%d]' % index, seen_fields) for index, fieldset in enumerate(obj.fieldsets))))

    def _check_fieldsets_item(self, obj, fieldset, label, seen_fields):
        """Check an item of `fieldsets`, i.e. check that this is a pair of a
        set name and a dictionary containing "fields" key."""
        if not isinstance(fieldset, (list, tuple)):
            return must_be('a list or tuple', option=label, obj=obj, id='admin.E008')
        elif len(fieldset) != 2:
            return must_be('of length 2', option=label, obj=obj, id='admin.E009')
        elif not isinstance(fieldset[1], dict):
            return must_be('a dictionary', option='%s[1]' % label, obj=obj, id='admin.E010')
        elif 'fields' not in fieldset[1]:
            return [checks.Error("The value of '%s[1]' must contain the key 'fields'." % label, obj=obj.__class__, id='admin.E011')]
        elif not isinstance(fieldset[1]['fields'], (list, tuple)):
            return must_be('a list or tuple', option="%s[1]['fields']" % label, obj=obj, id='admin.E008')
        seen_fields.extend(flatten(fieldset[1]['fields']))
        if len(seen_fields) != len(set(seen_fields)):
            return [checks.Error("There are duplicate field(s) in '%s[1]'." % label, obj=obj.__class__, id='admin.E012')]
        return list(chain.from_iterable((self._check_field_spec(obj, fieldset_fields, '%s[1]["fields"]' % label) for fieldset_fields in fieldset[1]['fields'])))

    def _check_field_spec(self, obj, fields, label):
        """`fields` should be an item of `fields` or an item of
        fieldset[1]['fields'] for any `fieldset` in `fieldsets`. It should be a
        field name or a tuple of field names."""
        if isinstance(fields, tuple):
            return list(chain.from_iterable((self._check_field_spec_item(obj, field_name, '%s[%d]' % (label, index)) for index, field_name in enumerate(fields))))
        else:
            return self._check_field_spec_item(obj, fields, label)

    def _check_field_spec_item(self, obj, field_name, label):
        if field_name in obj.readonly_fields:
            return []
        else:
            try:
                field = obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return []
            else:
                if isinstance(field, models.ManyToManyField) and (not field.remote_field.through._meta.auto_created):
                    return [checks.Error("The value of '%s' cannot include the ManyToManyField '%s', because that field manually specifies a relationship model." % (label, field_name), obj=obj.__class__, id='admin.E013')]
                else:
                    return []

    def _check_exclude(self, obj):
        """Check that exclude is a sequence without duplicates."""
        if obj.exclude is None:
            return []
        elif not isinstance(obj.exclude, (list, tuple)):
            return must_be('a list or tuple', option='exclude', obj=obj, id='admin.E014')
        elif len(obj.exclude) > len(set(obj.exclude)):
            return [checks.Error("The value of 'exclude' contains duplicate field(s).", obj=obj.__class__, id='admin.E015')]
        else:
            return []

    def _check_form(self, obj):
        """Check that form subclasses BaseModelForm."""
        if not _issubclass(obj.form, BaseModelForm):
            return must_inherit_from(parent='BaseModelForm', option='form', obj=obj, id='admin.E016')
        else:
            return []

    def _check_filter_vertical(self, obj):
        """Check that filter_vertical is a sequence of field names."""
        if not isinstance(obj.filter_vertical, (list, tuple)):
            return must_be('a list or tuple', option='filter_vertical', obj=obj, id='admin.E017')
        else:
            return list(chain.from_iterable((self._check_filter_item(obj, field_name, 'filter_vertical[%d]' % index) for index, field_name in enumerate(obj.filter_vertical))))

    def _check_filter_horizontal(self, obj):
        """Check that filter_horizontal is a sequence of field names."""
        if not isinstance(obj.filter_horizontal, (list, tuple)):
            return must_be('a list or tuple', option='filter_horizontal', obj=obj, id='admin.E018')
        else:
            return list(chain.from_iterable((self._check_filter_item(obj, field_name, 'filter_horizontal[%d]' % index) for index, field_name in enumerate(obj.filter_horizontal))))

    def _check_filter_item(self, obj, field_name, label):
        """Check one item of `filter_vertical` or `filter_horizontal`, i.e.
        check that given field exists and is a ManyToManyField."""
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E019')
        else:
            if not field.many_to_many or isinstance(field, models.ManyToManyRel):
                return must_be('a many-to-many field', option=label, obj=obj, id='admin.E020')
            elif not field.remote_field.through._meta.auto_created:
                return [checks.Error(f"The value of '{label}' cannot include the ManyToManyField '{field_name}', because that field manually specifies a relationship model.", obj=obj.__class__, id='admin.E013')]
            else:
                return []

    def _check_radio_fields(self, obj):
        """Check that `radio_fields` is a dictionary."""
        if not isinstance(obj.radio_fields, dict):
            return must_be('a dictionary', option='radio_fields', obj=obj, id='admin.E021')
        else:
            return list(chain.from_iterable((self._check_radio_fields_key(obj, field_name, 'radio_fields') + self._check_radio_fields_value(obj, val, 'radio_fields["%s"]' % field_name) for field_name, val in obj.radio_fields.items())))

    def _check_radio_fields_key(self, obj, field_name, label):
        """Check that a key of `radio_fields` dictionary is name of existing
        field and that the field is a ForeignKey or has `choices` defined."""
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E022')
        else:
            if not (isinstance(field, models.ForeignKey) or field.choices):
                return [checks.Error("The value of '%s' refers to '%s', which is not an instance of ForeignKey, and does not have a 'choices' definition." % (label, field_name), obj=obj.__class__, id='admin.E023')]
            else:
                return []

    def _check_radio_fields_value(self, obj, val, label):
        """Check type of a value of `radio_fields` dictionary."""
        from django.contrib.admin.options import HORIZONTAL, VERTICAL
        if val not in (HORIZONTAL, VERTICAL):
            return [checks.Error("The value of '%s' must be either admin.HORIZONTAL or admin.VERTICAL." % label, obj=obj.__class__, id='admin.E024')]
        else:
            return []

    def _check_view_on_site_url(self, obj):
        if not callable(obj.view_on_site) and (not isinstance(obj.view_on_site, bool)):
            return [checks.Error("The value of 'view_on_site' must be a callable or a boolean value.", obj=obj.__class__, id='admin.E025')]
        else:
            return []

    def _check_prepopulated_fields(self, obj):
        """Check that `prepopulated_fields` is a dictionary containing allowed
        field types."""
        if not isinstance(obj.prepopulated_fields, dict):
            return must_be('a dictionary', option='prepopulated_fields', obj=obj, id='admin.E026')
        else:
            return list(chain.from_iterable((self._check_prepopulated_fields_key(obj, field_name, 'prepopulated_fields') + self._check_prepopulated_fields_value(obj, val, 'prepopulated_fields["%s"]' % field_name) for field_name, val in obj.prepopulated_fields.items())))

    def _check_prepopulated_fields_key(self, obj, field_name, label):
        """Check a key of `prepopulated_fields` dictionary, i.e. check that it
        is a name of existing field and the field is one of the allowed types.
        """
        try:
            field = obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E027')
        else:
            if isinstance(field, (models.DateTimeField, models.ForeignKey, models.ManyToManyField)):
                return [checks.Error("The value of '%s' refers to '%s', which must not be a DateTimeField, a ForeignKey, a OneToOneField, or a ManyToManyField." % (label, field_name), obj=obj.__class__, id='admin.E028')]
            else:
                return []

    def _check_prepopulated_fields_value(self, obj, val, label):
        """Check a value of `prepopulated_fields` dictionary, i.e. it's an
        iterable of existing fields."""
        if not isinstance(val, (list, tuple)):
            return must_be('a list or tuple', option=label, obj=obj, id='admin.E029')
        else:
            return list(chain.from_iterable((self._check_prepopulated_fields_value_item(obj, subfield_name, '%s[%r]' % (label, index)) for index, subfield_name in enumerate(val))))

    def _check_prepopulated_fields_value_item(self, obj, field_name, label):
        """For `prepopulated_fields` equal to {"slug": ("title",)},
        `field_name` is "title"."""
        try:
            obj.model._meta.get_field(field_name)
        except FieldDoesNotExist:
            return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E030')
        else:
            return []

    def _check_ordering(self, obj):
        """Check that ordering refers to existing fields or is random."""
        if obj.ordering is None:
            return []
        elif not isinstance(obj.ordering, (list, tuple)):
            return must_be('a list or tuple', option='ordering', obj=obj, id='admin.E031')
        else:
            return list(chain.from_iterable((self._check_ordering_item(obj, field_name, 'ordering[%d]' % index) for index, field_name in enumerate(obj.ordering))))

    def _check_ordering_item(self, obj, field_name, label):
        """Check that `ordering` refers to existing fields."""
        if isinstance(field_name, (Combinable, models.OrderBy)):
            if not isinstance(field_name, models.OrderBy):
                field_name = field_name.asc()
            if isinstance(field_name.expression, models.F):
                field_name = field_name.expression.name
            else:
                return []
        if field_name == '?' and len(obj.ordering) != 1:
            return [checks.Error("The value of 'ordering' has the random ordering marker '?', but contains other fields as well.", hint='Either remove the "?", or remove the other fields.', obj=obj.__class__, id='admin.E032')]
        elif field_name == '?':
            return []
        elif LOOKUP_SEP in field_name:
            return []
        else:
            field_name = field_name.removeprefix('-')
            if field_name == 'pk':
                return []
            try:
                obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return refer_to_missing_field(field=field_name, option=label, obj=obj, id='admin.E033')
            else:
                return []

    def _check_readonly_fields(self, obj):
        """Check that readonly_fields refers to proper attribute or field."""
        if obj.readonly_fields == ():
            return []
        elif not isinstance(obj.readonly_fields, (list, tuple)):
            return must_be('a list or tuple', option='readonly_fields', obj=obj, id='admin.E034')
        else:
            return list(chain.from_iterable((self._check_readonly_fields_item(obj, field_name, 'readonly_fields[%d]' % index) for index, field_name in enumerate(obj.readonly_fields))))

    def _check_readonly_fields_item(self, obj, field_name, label):
        if callable(field_name):
            return []
        elif hasattr(obj, field_name):
            return []
        elif hasattr(obj.model, field_name):
            return []
        else:
            try:
                obj.model._meta.get_field(field_name)
            except FieldDoesNotExist:
                return [checks.Error("The value of '%s' refers to '%s', which is not a callable, an attribute of '%s', or an attribute of '%s'." % (label, field_name, obj.__class__.__name__, obj.model._meta.label), obj=obj.__class__, id='admin.E035')]
            else:
                return []