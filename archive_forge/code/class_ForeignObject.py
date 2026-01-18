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
class ForeignObject(RelatedField):
    """
    Abstraction of the ForeignKey relation to support multi-column relations.
    """
    many_to_many = False
    many_to_one = True
    one_to_many = False
    one_to_one = False
    requires_unique_target = True
    related_accessor_class = ReverseManyToOneDescriptor
    forward_related_accessor_class = ForwardManyToOneDescriptor
    rel_class = ForeignObjectRel

    def __init__(self, to, on_delete, from_fields, to_fields, rel=None, related_name=None, related_query_name=None, limit_choices_to=None, parent_link=False, swappable=True, **kwargs):
        if rel is None:
            rel = self.rel_class(self, to, related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to, parent_link=parent_link, on_delete=on_delete)
        super().__init__(rel=rel, related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to, **kwargs)
        self.from_fields = from_fields
        self.to_fields = to_fields
        self.swappable = swappable

    def __copy__(self):
        obj = super().__copy__()
        obj.__dict__.pop('path_infos', None)
        obj.__dict__.pop('reverse_path_infos', None)
        return obj

    def check(self, **kwargs):
        return [*super().check(**kwargs), *self._check_to_fields_exist(), *self._check_unique_target()]

    def _check_to_fields_exist(self):
        if isinstance(self.remote_field.model, str):
            return []
        errors = []
        for to_field in self.to_fields:
            if to_field:
                try:
                    self.remote_field.model._meta.get_field(to_field)
                except exceptions.FieldDoesNotExist:
                    errors.append(checks.Error("The to_field '%s' doesn't exist on the related model '%s'." % (to_field, self.remote_field.model._meta.label), obj=self, id='fields.E312'))
        return errors

    def _check_unique_target(self):
        rel_is_string = isinstance(self.remote_field.model, str)
        if rel_is_string or not self.requires_unique_target:
            return []
        try:
            self.foreign_related_fields
        except exceptions.FieldDoesNotExist:
            return []
        if not self.foreign_related_fields:
            return []
        unique_foreign_fields = {frozenset([f.name]) for f in self.remote_field.model._meta.get_fields() if getattr(f, 'unique', False)}
        unique_foreign_fields.update({frozenset(ut) for ut in self.remote_field.model._meta.unique_together})
        unique_foreign_fields.update({frozenset(uc.fields) for uc in self.remote_field.model._meta.total_unique_constraints})
        foreign_fields = {f.name for f in self.foreign_related_fields}
        has_unique_constraint = any((u <= foreign_fields for u in unique_foreign_fields))
        if not has_unique_constraint and len(self.foreign_related_fields) > 1:
            field_combination = ', '.join(("'%s'" % rel_field.name for rel_field in self.foreign_related_fields))
            model_name = self.remote_field.model.__name__
            return [checks.Error("No subset of the fields %s on model '%s' is unique." % (field_combination, model_name), hint='Mark a single field as unique=True or add a set of fields to a unique constraint (via unique_together or a UniqueConstraint (without condition) in the model Meta.constraints).', obj=self, id='fields.E310')]
        elif not has_unique_constraint:
            field_name = self.foreign_related_fields[0].name
            model_name = self.remote_field.model.__name__
            return [checks.Error("'%s.%s' must be unique because it is referenced by a foreign key." % (model_name, field_name), hint='Add unique=True to this field or add a UniqueConstraint (without condition) in the model Meta.constraints.', obj=self, id='fields.E311')]
        else:
            return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['on_delete'] = self.remote_field.on_delete
        kwargs['from_fields'] = self.from_fields
        kwargs['to_fields'] = self.to_fields
        if self.remote_field.parent_link:
            kwargs['parent_link'] = self.remote_field.parent_link
        if isinstance(self.remote_field.model, str):
            if '.' in self.remote_field.model:
                app_label, model_name = self.remote_field.model.split('.')
                kwargs['to'] = '%s.%s' % (app_label, model_name.lower())
            else:
                kwargs['to'] = self.remote_field.model.lower()
        else:
            kwargs['to'] = self.remote_field.model._meta.label_lower
        swappable_setting = self.swappable_setting
        if swappable_setting is not None:
            if hasattr(kwargs['to'], 'setting_name'):
                if kwargs['to'].setting_name != swappable_setting:
                    raise ValueError('Cannot deconstruct a ForeignKey pointing to a model that is swapped in place of more than one model (%s and %s)' % (kwargs['to'].setting_name, swappable_setting))
            kwargs['to'] = SettingsReference(kwargs['to'], swappable_setting)
        return (name, path, args, kwargs)

    def resolve_related_fields(self):
        if not self.from_fields or len(self.from_fields) != len(self.to_fields):
            raise ValueError('Foreign Object from and to fields must be the same non-zero length')
        if isinstance(self.remote_field.model, str):
            raise ValueError('Related model %r cannot be resolved' % self.remote_field.model)
        related_fields = []
        for index in range(len(self.from_fields)):
            from_field_name = self.from_fields[index]
            to_field_name = self.to_fields[index]
            from_field = self if from_field_name == RECURSIVE_RELATIONSHIP_CONSTANT else self.opts.get_field(from_field_name)
            to_field = self.remote_field.model._meta.pk if to_field_name is None else self.remote_field.model._meta.get_field(to_field_name)
            related_fields.append((from_field, to_field))
        return related_fields

    @cached_property
    def related_fields(self):
        return self.resolve_related_fields()

    @cached_property
    def reverse_related_fields(self):
        return [(rhs_field, lhs_field) for lhs_field, rhs_field in self.related_fields]

    @cached_property
    def local_related_fields(self):
        return tuple((lhs_field for lhs_field, rhs_field in self.related_fields))

    @cached_property
    def foreign_related_fields(self):
        return tuple((rhs_field for lhs_field, rhs_field in self.related_fields if rhs_field))

    def get_local_related_value(self, instance):
        return self.get_instance_value_for_fields(instance, self.local_related_fields)

    def get_foreign_related_value(self, instance):
        return self.get_instance_value_for_fields(instance, self.foreign_related_fields)

    @staticmethod
    def get_instance_value_for_fields(instance, fields):
        ret = []
        opts = instance._meta
        for field in fields:
            if field.primary_key:
                possible_parent_link = opts.get_ancestor_link(field.model)
                if not possible_parent_link or possible_parent_link.primary_key or possible_parent_link.model._meta.abstract:
                    ret.append(instance.pk)
                    continue
            ret.append(getattr(instance, field.attname))
        return tuple(ret)

    def get_attname_column(self):
        attname, column = super().get_attname_column()
        return (attname, None)

    def get_joining_columns(self, reverse_join=False):
        warnings.warn('ForeignObject.get_joining_columns() is deprecated. Use get_joining_fields() instead.', RemovedInDjango60Warning)
        source = self.reverse_related_fields if reverse_join else self.related_fields
        return tuple(((lhs_field.column, rhs_field.column) for lhs_field, rhs_field in source))

    def get_reverse_joining_columns(self):
        warnings.warn('ForeignObject.get_reverse_joining_columns() is deprecated. Use get_reverse_joining_fields() instead.', RemovedInDjango60Warning)
        return self.get_joining_columns(reverse_join=True)

    def get_joining_fields(self, reverse_join=False):
        return tuple(self.reverse_related_fields if reverse_join else self.related_fields)

    def get_reverse_joining_fields(self):
        return self.get_joining_fields(reverse_join=True)

    def get_extra_descriptor_filter(self, instance):
        """
        Return an extra filter condition for related object fetching when
        user does 'instance.fieldname', that is the extra filter is used in
        the descriptor of the field.

        The filter should be either a dict usable in .filter(**kwargs) call or
        a Q-object. The condition will be ANDed together with the relation's
        joining columns.

        A parallel method is get_extra_restriction() which is used in
        JOIN and subquery conditions.
        """
        return {}

    def get_extra_restriction(self, alias, related_alias):
        """
        Return a pair condition used for joining and subquery pushdown. The
        condition is something that responds to as_sql(compiler, connection)
        method.

        Note that currently referring both the 'alias' and 'related_alias'
        will not work in some conditions, like subquery pushdown.

        A parallel method is get_extra_descriptor_filter() which is used in
        instance.fieldname related object fetching.
        """
        return None

    def get_path_info(self, filtered_relation=None):
        """Get path from this field to the related model."""
        opts = self.remote_field.model._meta
        from_opts = self.model._meta
        return [PathInfo(from_opts=from_opts, to_opts=opts, target_fields=self.foreign_related_fields, join_field=self, m2m=False, direct=True, filtered_relation=filtered_relation)]

    @cached_property
    def path_infos(self):
        return self.get_path_info()

    def get_reverse_path_info(self, filtered_relation=None):
        """Get path from the related model to this field's model."""
        opts = self.model._meta
        from_opts = self.remote_field.model._meta
        return [PathInfo(from_opts=from_opts, to_opts=opts, target_fields=(opts.pk,), join_field=self.remote_field, m2m=not self.unique, direct=False, filtered_relation=filtered_relation)]

    @cached_property
    def reverse_path_infos(self):
        return self.get_reverse_path_info()

    @classmethod
    @functools.cache
    def get_class_lookups(cls):
        bases = inspect.getmro(cls)
        bases = bases[:bases.index(ForeignObject) + 1]
        class_lookups = [parent.__dict__.get('class_lookups', {}) for parent in bases]
        return cls.merge_dicts(class_lookups)

    def contribute_to_class(self, cls, name, private_only=False, **kwargs):
        super().contribute_to_class(cls, name, private_only=private_only, **kwargs)
        setattr(cls, self.name, self.forward_related_accessor_class(self))

    def contribute_to_related_class(self, cls, related):
        if not self.remote_field.is_hidden() and (not related.related_model._meta.swapped):
            setattr(cls._meta.concrete_model, related.get_accessor_name(), self.related_accessor_class(related))
            if self.remote_field.limit_choices_to:
                cls._meta.related_fkey_lookups.append(self.remote_field.limit_choices_to)