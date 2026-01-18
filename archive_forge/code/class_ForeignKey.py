import functools
import inspect
import warnings
from functools import partial
from django import forms
from django.apps import apps
from django.conf import SettingsReference, settings
from django.core import checks, exceptions
from django.db import connection, router
from django.db.backends import utils
from django.db.models import Q
from django.db.models.constants import LOOKUP_SEP
from django.db.models.deletion import CASCADE, SET_DEFAULT, SET_NULL
from django.db.models.query_utils import PathInfo
from django.db.models.utils import make_model_tuple
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.translation import gettext_lazy as _
from . import Field
from .mixins import FieldCacheMixin
from .related_descriptors import (
from .related_lookups import (
from .reverse_related import ForeignObjectRel, ManyToManyRel, ManyToOneRel, OneToOneRel
class ForeignKey(ForeignObject):
    """
    Provide a many-to-one relation by adding a column to the local model
    to hold the remote value.

    By default ForeignKey will target the pk of the remote model but this
    behavior can be changed by using the ``to_field`` argument.
    """
    descriptor_class = ForeignKeyDeferredAttribute
    many_to_many = False
    many_to_one = True
    one_to_many = False
    one_to_one = False
    rel_class = ManyToOneRel
    empty_strings_allowed = False
    default_error_messages = {'invalid': _('%(model)s instance with %(field)s %(value)r does not exist.')}
    description = _('Foreign Key (type determined by related field)')

    def __init__(self, to, on_delete, related_name=None, related_query_name=None, limit_choices_to=None, parent_link=False, to_field=None, db_constraint=True, **kwargs):
        try:
            to._meta.model_name
        except AttributeError:
            if not isinstance(to, str):
                raise TypeError('%s(%r) is invalid. First parameter to ForeignKey must be either a model, a model name, or the string %r' % (self.__class__.__name__, to, RECURSIVE_RELATIONSHIP_CONSTANT))
        else:
            to_field = to_field or (to._meta.pk and to._meta.pk.name)
        if not callable(on_delete):
            raise TypeError('on_delete must be callable.')
        kwargs['rel'] = self.rel_class(self, to, to_field, related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to, parent_link=parent_link, on_delete=on_delete)
        kwargs.setdefault('db_index', True)
        super().__init__(to, on_delete, related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to, from_fields=[RECURSIVE_RELATIONSHIP_CONSTANT], to_fields=[to_field], **kwargs)
        self.db_constraint = db_constraint

    def __class_getitem__(cls, *args, **kwargs):
        return cls

    def check(self, **kwargs):
        return [*super().check(**kwargs), *self._check_on_delete(), *self._check_unique()]

    def _check_on_delete(self):
        on_delete = getattr(self.remote_field, 'on_delete', None)
        if on_delete == SET_NULL and (not self.null):
            return [checks.Error('Field specifies on_delete=SET_NULL, but cannot be null.', hint='Set null=True argument on the field, or change the on_delete rule.', obj=self, id='fields.E320')]
        elif on_delete == SET_DEFAULT and (not self.has_default()):
            return [checks.Error('Field specifies on_delete=SET_DEFAULT, but has no default value.', hint='Set a default value, or change the on_delete rule.', obj=self, id='fields.E321')]
        else:
            return []

    def _check_unique(self, **kwargs):
        return [checks.Warning('Setting unique=True on a ForeignKey has the same effect as using a OneToOneField.', hint='ForeignKey(unique=True) is usually better served by a OneToOneField.', obj=self, id='fields.W342')] if self.unique else []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        del kwargs['to_fields']
        del kwargs['from_fields']
        if self.db_index:
            del kwargs['db_index']
        else:
            kwargs['db_index'] = False
        if self.db_constraint is not True:
            kwargs['db_constraint'] = self.db_constraint
        to_meta = getattr(self.remote_field.model, '_meta', None)
        if self.remote_field.field_name and (not to_meta or (to_meta.pk and self.remote_field.field_name != to_meta.pk.name)):
            kwargs['to_field'] = self.remote_field.field_name
        return (name, path, args, kwargs)

    def to_python(self, value):
        return self.target_field.to_python(value)

    @property
    def target_field(self):
        return self.foreign_related_fields[0]

    def validate(self, value, model_instance):
        if self.remote_field.parent_link:
            return
        super().validate(value, model_instance)
        if value is None:
            return
        using = router.db_for_read(self.remote_field.model, instance=model_instance)
        qs = self.remote_field.model._base_manager.using(using).filter(**{self.remote_field.field_name: value})
        qs = qs.complex_filter(self.get_limit_choices_to())
        if not qs.exists():
            raise exceptions.ValidationError(self.error_messages['invalid'], code='invalid', params={'model': self.remote_field.model._meta.verbose_name, 'pk': value, 'field': self.remote_field.field_name, 'value': value})

    def resolve_related_fields(self):
        related_fields = super().resolve_related_fields()
        for from_field, to_field in related_fields:
            if to_field and to_field.model != self.remote_field.model._meta.concrete_model:
                raise exceptions.FieldError("'%s.%s' refers to field '%s' which is not local to model '%s'." % (self.model._meta.label, self.name, to_field.name, self.remote_field.model._meta.concrete_model._meta.label))
        return related_fields

    def get_attname(self):
        return '%s_id' % self.name

    def get_attname_column(self):
        attname = self.get_attname()
        column = self.db_column or attname
        return (attname, column)

    def get_default(self):
        """Return the to_field if the default value is an object."""
        field_default = super().get_default()
        if isinstance(field_default, self.remote_field.model):
            return getattr(field_default, self.target_field.attname)
        return field_default

    def get_db_prep_save(self, value, connection):
        if value is None or (value == '' and (not self.target_field.empty_strings_allowed or connection.features.interprets_empty_strings_as_nulls)):
            return None
        else:
            return self.target_field.get_db_prep_save(value, connection=connection)

    def get_db_prep_value(self, value, connection, prepared=False):
        return self.target_field.get_db_prep_value(value, connection, prepared)

    def get_prep_value(self, value):
        return self.target_field.get_prep_value(value)

    def contribute_to_related_class(self, cls, related):
        super().contribute_to_related_class(cls, related)
        if self.remote_field.field_name is None:
            self.remote_field.field_name = cls._meta.pk.name

    def formfield(self, *, using=None, **kwargs):
        if isinstance(self.remote_field.model, str):
            raise ValueError('Cannot create form field for %r yet, because its related model %r has not been loaded yet' % (self.name, self.remote_field.model))
        return super().formfield(**{'form_class': forms.ModelChoiceField, 'queryset': self.remote_field.model._default_manager.using(using), 'to_field_name': self.remote_field.field_name, **kwargs, 'blank': self.blank})

    def db_check(self, connection):
        return None

    def db_type(self, connection):
        return self.target_field.rel_db_type(connection=connection)

    def cast_db_type(self, connection):
        return self.target_field.cast_db_type(connection=connection)

    def db_parameters(self, connection):
        target_db_parameters = self.target_field.db_parameters(connection)
        return {'type': self.db_type(connection), 'check': self.db_check(connection), 'collation': target_db_parameters.get('collation')}

    def convert_empty_strings(self, value, expression, connection):
        if not value and isinstance(value, str):
            return None
        return value

    def get_db_converters(self, connection):
        converters = super().get_db_converters(connection)
        if connection.features.interprets_empty_strings_as_nulls:
            converters += [self.convert_empty_strings]
        return converters

    def get_col(self, alias, output_field=None):
        if output_field is None:
            output_field = self.target_field
            while isinstance(output_field, ForeignKey):
                output_field = output_field.target_field
                if output_field is self:
                    raise ValueError('Cannot resolve output_field.')
        return super().get_col(alias, output_field)