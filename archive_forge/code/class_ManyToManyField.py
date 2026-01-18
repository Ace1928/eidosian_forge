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
class ManyToManyField(RelatedField):
    """
    Provide a many-to-many relation by using an intermediary model that
    holds two ForeignKey fields pointed at the two sides of the relation.

    Unless a ``through`` model was provided, ManyToManyField will use the
    create_many_to_many_intermediary_model factory to automatically generate
    the intermediary model.
    """
    many_to_many = True
    many_to_one = False
    one_to_many = False
    one_to_one = False
    rel_class = ManyToManyRel
    description = _('Many-to-many relationship')

    def __init__(self, to, related_name=None, related_query_name=None, limit_choices_to=None, symmetrical=None, through=None, through_fields=None, db_constraint=True, db_table=None, swappable=True, **kwargs):
        try:
            to._meta
        except AttributeError:
            if not isinstance(to, str):
                raise TypeError('%s(%r) is invalid. First parameter to ManyToManyField must be either a model, a model name, or the string %r' % (self.__class__.__name__, to, RECURSIVE_RELATIONSHIP_CONSTANT))
        if symmetrical is None:
            symmetrical = to == RECURSIVE_RELATIONSHIP_CONSTANT
        if through is not None and db_table is not None:
            raise ValueError('Cannot specify a db_table if an intermediary model is used.')
        kwargs['rel'] = self.rel_class(self, to, related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to, symmetrical=symmetrical, through=through, through_fields=through_fields, db_constraint=db_constraint)
        self.has_null_arg = 'null' in kwargs
        super().__init__(related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to, **kwargs)
        self.db_table = db_table
        self.swappable = swappable

    def check(self, **kwargs):
        return [*super().check(**kwargs), *self._check_unique(**kwargs), *self._check_relationship_model(**kwargs), *self._check_ignored_options(**kwargs), *self._check_table_uniqueness(**kwargs)]

    def _check_unique(self, **kwargs):
        if self.unique:
            return [checks.Error('ManyToManyFields cannot be unique.', obj=self, id='fields.E330')]
        return []

    def _check_ignored_options(self, **kwargs):
        warnings = []
        if self.has_null_arg:
            warnings.append(checks.Warning('null has no effect on ManyToManyField.', obj=self, id='fields.W340'))
        if self._validators:
            warnings.append(checks.Warning('ManyToManyField does not support validators.', obj=self, id='fields.W341'))
        if self.remote_field.symmetrical and self._related_name:
            warnings.append(checks.Warning('related_name has no effect on ManyToManyField with a symmetrical relationship, e.g. to "self".', obj=self, id='fields.W345'))
        if self.db_comment:
            warnings.append(checks.Warning('db_comment has no effect on ManyToManyField.', obj=self, id='fields.W346'))
        return warnings

    def _check_relationship_model(self, from_model=None, **kwargs):
        if hasattr(self.remote_field.through, '_meta'):
            qualified_model_name = '%s.%s' % (self.remote_field.through._meta.app_label, self.remote_field.through.__name__)
        else:
            qualified_model_name = self.remote_field.through
        errors = []
        if self.remote_field.through not in self.opts.apps.get_models(include_auto_created=True):
            errors.append(checks.Error("Field specifies a many-to-many relation through model '%s', which has not been installed." % qualified_model_name, obj=self, id='fields.E331'))
        else:
            assert from_model is not None, "ManyToManyField with intermediate tables cannot be checked if you don't pass the model where the field is attached to."
            to_model = resolve_relation(from_model, self.remote_field.model)
            from_model_name = from_model._meta.object_name
            if isinstance(to_model, str):
                to_model_name = to_model
            else:
                to_model_name = to_model._meta.object_name
            relationship_model_name = self.remote_field.through._meta.object_name
            self_referential = from_model == to_model
            if self_referential:
                seen_self = sum((from_model == getattr(field.remote_field, 'model', None) for field in self.remote_field.through._meta.fields))
                if seen_self > 2 and (not self.remote_field.through_fields):
                    errors.append(checks.Error("The model is used as an intermediate model by '%s', but it has more than two foreign keys to '%s', which is ambiguous. You must specify which two foreign keys Django should use via the through_fields keyword argument." % (self, from_model_name), hint='Use through_fields to specify which two foreign keys Django should use.', obj=self.remote_field.through, id='fields.E333'))
            else:
                seen_from = sum((from_model == getattr(field.remote_field, 'model', None) for field in self.remote_field.through._meta.fields))
                seen_to = sum((to_model == getattr(field.remote_field, 'model', None) for field in self.remote_field.through._meta.fields))
                if seen_from > 1 and (not self.remote_field.through_fields):
                    errors.append(checks.Error("The model is used as an intermediate model by '%s', but it has more than one foreign key from '%s', which is ambiguous. You must specify which foreign key Django should use via the through_fields keyword argument." % (self, from_model_name), hint='If you want to create a recursive relationship, use ManyToManyField("%s", through="%s").' % (RECURSIVE_RELATIONSHIP_CONSTANT, relationship_model_name), obj=self, id='fields.E334'))
                if seen_to > 1 and (not self.remote_field.through_fields):
                    errors.append(checks.Error("The model is used as an intermediate model by '%s', but it has more than one foreign key to '%s', which is ambiguous. You must specify which foreign key Django should use via the through_fields keyword argument." % (self, to_model_name), hint='If you want to create a recursive relationship, use ManyToManyField("%s", through="%s").' % (RECURSIVE_RELATIONSHIP_CONSTANT, relationship_model_name), obj=self, id='fields.E335'))
                if seen_from == 0 or seen_to == 0:
                    errors.append(checks.Error("The model is used as an intermediate model by '%s', but it does not have a foreign key to '%s' or '%s'." % (self, from_model_name, to_model_name), obj=self.remote_field.through, id='fields.E336'))
        if self.remote_field.through_fields is not None:
            if not (len(self.remote_field.through_fields) >= 2 and self.remote_field.through_fields[0] and self.remote_field.through_fields[1]):
                errors.append(checks.Error("Field specifies 'through_fields' but does not provide the names of the two link fields that should be used for the relation through model '%s'." % qualified_model_name, hint="Make sure you specify 'through_fields' as through_fields=('field1', 'field2')", obj=self, id='fields.E337'))
            else:
                assert from_model is not None, "ManyToManyField with intermediate tables cannot be checked if you don't pass the model where the field is attached to."
                source, through, target = (from_model, self.remote_field.through, self.remote_field.model)
                source_field_name, target_field_name = self.remote_field.through_fields[:2]
                for field_name, related_model in ((source_field_name, source), (target_field_name, target)):
                    possible_field_names = []
                    for f in through._meta.fields:
                        if hasattr(f, 'remote_field') and getattr(f.remote_field, 'model', None) == related_model:
                            possible_field_names.append(f.name)
                    if possible_field_names:
                        hint = "Did you mean one of the following foreign keys to '%s': %s?" % (related_model._meta.object_name, ', '.join(possible_field_names))
                    else:
                        hint = None
                    try:
                        field = through._meta.get_field(field_name)
                    except exceptions.FieldDoesNotExist:
                        errors.append(checks.Error("The intermediary model '%s' has no field '%s'." % (qualified_model_name, field_name), hint=hint, obj=self, id='fields.E338'))
                    else:
                        if not (hasattr(field, 'remote_field') and getattr(field.remote_field, 'model', None) == related_model):
                            errors.append(checks.Error("'%s.%s' is not a foreign key to '%s'." % (through._meta.object_name, field_name, related_model._meta.object_name), hint=hint, obj=self, id='fields.E339'))
        return errors

    def _check_table_uniqueness(self, **kwargs):
        if isinstance(self.remote_field.through, str) or not self.remote_field.through._meta.managed:
            return []
        registered_tables = {model._meta.db_table: model for model in self.opts.apps.get_models(include_auto_created=True) if model != self.remote_field.through and model._meta.managed}
        m2m_db_table = self.m2m_db_table()
        model = registered_tables.get(m2m_db_table)
        if model and model._meta.concrete_model != self.remote_field.through._meta.concrete_model:
            if model._meta.auto_created:

                def _get_field_name(model):
                    for field in model._meta.auto_created._meta.many_to_many:
                        if field.remote_field.through is model:
                            return field.name
                opts = model._meta.auto_created._meta
                clashing_obj = '%s.%s' % (opts.label, _get_field_name(model))
            else:
                clashing_obj = model._meta.label
            if settings.DATABASE_ROUTERS:
                error_class, error_id = (checks.Warning, 'fields.W344')
                error_hint = 'You have configured settings.DATABASE_ROUTERS. Verify that the table of %r is correctly routed to a separate database.' % clashing_obj
            else:
                error_class, error_id = (checks.Error, 'fields.E340')
                error_hint = None
            return [error_class("The field's intermediary table '%s' clashes with the table name of '%s'." % (m2m_db_table, clashing_obj), obj=self, hint=error_hint, id=error_id)]
        return []

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        if self.db_table is not None:
            kwargs['db_table'] = self.db_table
        if self.remote_field.db_constraint is not True:
            kwargs['db_constraint'] = self.remote_field.db_constraint
        if isinstance(self.remote_field.model, str):
            if '.' in self.remote_field.model:
                app_label, model_name = self.remote_field.model.split('.')
                kwargs['to'] = '%s.%s' % (app_label, model_name.lower())
            else:
                kwargs['to'] = self.remote_field.model.lower()
        else:
            kwargs['to'] = self.remote_field.model._meta.label_lower
        if getattr(self.remote_field, 'through', None) is not None:
            if isinstance(self.remote_field.through, str):
                kwargs['through'] = self.remote_field.through
            elif not self.remote_field.through._meta.auto_created:
                kwargs['through'] = self.remote_field.through._meta.label
        swappable_setting = self.swappable_setting
        if swappable_setting is not None:
            if hasattr(kwargs['to'], 'setting_name'):
                if kwargs['to'].setting_name != swappable_setting:
                    raise ValueError('Cannot deconstruct a ManyToManyField pointing to a model that is swapped in place of more than one model (%s and %s)' % (kwargs['to'].setting_name, swappable_setting))
            kwargs['to'] = SettingsReference(kwargs['to'], swappable_setting)
        return (name, path, args, kwargs)

    def _get_path_info(self, direct=False, filtered_relation=None):
        """Called by both direct and indirect m2m traversal."""
        int_model = self.remote_field.through
        linkfield1 = int_model._meta.get_field(self.m2m_field_name())
        linkfield2 = int_model._meta.get_field(self.m2m_reverse_field_name())
        if direct:
            join1infos = linkfield1.reverse_path_infos
            if filtered_relation:
                join2infos = linkfield2.get_path_info(filtered_relation)
            else:
                join2infos = linkfield2.path_infos
        else:
            join1infos = linkfield2.reverse_path_infos
            if filtered_relation:
                join2infos = linkfield1.get_path_info(filtered_relation)
            else:
                join2infos = linkfield1.path_infos
        join1_final = join1infos[-1].to_opts
        join2_initial = join2infos[0].from_opts
        if join1_final is join2_initial:
            intermediate_infos = []
        elif issubclass(join1_final.model, join2_initial.model):
            intermediate_infos = join1_final.get_path_to_parent(join2_initial.model)
        else:
            intermediate_infos = join2_initial.get_path_from_parent(join1_final.model)
        return [*join1infos, *intermediate_infos, *join2infos]

    def get_path_info(self, filtered_relation=None):
        return self._get_path_info(direct=True, filtered_relation=filtered_relation)

    @cached_property
    def path_infos(self):
        return self.get_path_info()

    def get_reverse_path_info(self, filtered_relation=None):
        return self._get_path_info(direct=False, filtered_relation=filtered_relation)

    @cached_property
    def reverse_path_infos(self):
        return self.get_reverse_path_info()

    def _get_m2m_db_table(self, opts):
        """
        Function that can be curried to provide the m2m table name for this
        relation.
        """
        if self.remote_field.through is not None:
            return self.remote_field.through._meta.db_table
        elif self.db_table:
            return self.db_table
        else:
            m2m_table_name = '%s_%s' % (utils.strip_quotes(opts.db_table), self.name)
            return utils.truncate_name(m2m_table_name, connection.ops.max_name_length())

    def _get_m2m_attr(self, related, attr):
        """
        Function that can be curried to provide the source accessor or DB
        column name for the m2m table.
        """
        cache_attr = '_m2m_%s_cache' % attr
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)
        if self.remote_field.through_fields is not None:
            link_field_name = self.remote_field.through_fields[0]
        else:
            link_field_name = None
        for f in self.remote_field.through._meta.fields:
            if f.is_relation and f.remote_field.model == related.related_model and (link_field_name is None or link_field_name == f.name):
                setattr(self, cache_attr, getattr(f, attr))
                return getattr(self, cache_attr)

    def _get_m2m_reverse_attr(self, related, attr):
        """
        Function that can be curried to provide the related accessor or DB
        column name for the m2m table.
        """
        cache_attr = '_m2m_reverse_%s_cache' % attr
        if hasattr(self, cache_attr):
            return getattr(self, cache_attr)
        found = False
        if self.remote_field.through_fields is not None:
            link_field_name = self.remote_field.through_fields[1]
        else:
            link_field_name = None
        for f in self.remote_field.through._meta.fields:
            if f.is_relation and f.remote_field.model == related.model:
                if link_field_name is None and related.related_model == related.model:
                    if found:
                        setattr(self, cache_attr, getattr(f, attr))
                        break
                    else:
                        found = True
                elif link_field_name is None or link_field_name == f.name:
                    setattr(self, cache_attr, getattr(f, attr))
                    break
        return getattr(self, cache_attr)

    def contribute_to_class(self, cls, name, **kwargs):
        if self.remote_field.symmetrical and (self.remote_field.model == RECURSIVE_RELATIONSHIP_CONSTANT or self.remote_field.model == cls._meta.object_name):
            self.remote_field.related_name = '%s_rel_+' % name
        elif self.remote_field.is_hidden():
            self.remote_field.related_name = '_%s_%s_%s_+' % (cls._meta.app_label, cls.__name__.lower(), name)
        super().contribute_to_class(cls, name, **kwargs)
        if not cls._meta.abstract:
            if self.remote_field.through:

                def resolve_through_model(_, model, field):
                    field.remote_field.through = model
                lazy_related_operation(resolve_through_model, cls, self.remote_field.through, field=self)
            elif not cls._meta.swapped:
                self.remote_field.through = create_many_to_many_intermediary_model(self, cls)
        setattr(cls, self.name, ManyToManyDescriptor(self.remote_field, reverse=False))
        self.m2m_db_table = partial(self._get_m2m_db_table, cls._meta)

    def contribute_to_related_class(self, cls, related):
        if not self.remote_field.is_hidden() and (not related.related_model._meta.swapped):
            setattr(cls, related.get_accessor_name(), ManyToManyDescriptor(self.remote_field, reverse=True))
        self.m2m_column_name = partial(self._get_m2m_attr, related, 'column')
        self.m2m_reverse_name = partial(self._get_m2m_reverse_attr, related, 'column')
        self.m2m_field_name = partial(self._get_m2m_attr, related, 'name')
        self.m2m_reverse_field_name = partial(self._get_m2m_reverse_attr, related, 'name')
        get_m2m_rel = partial(self._get_m2m_attr, related, 'remote_field')
        self.m2m_target_field_name = lambda: get_m2m_rel().field_name
        get_m2m_reverse_rel = partial(self._get_m2m_reverse_attr, related, 'remote_field')
        self.m2m_reverse_target_field_name = lambda: get_m2m_reverse_rel().field_name

    def set_attributes_from_rel(self):
        pass

    def value_from_object(self, obj):
        return [] if obj.pk is None else list(getattr(obj, self.attname).all())

    def save_form_data(self, instance, data):
        getattr(instance, self.attname).set(data)

    def formfield(self, *, using=None, **kwargs):
        defaults = {'form_class': forms.ModelMultipleChoiceField, 'queryset': self.remote_field.model._default_manager.using(using), **kwargs}
        if defaults.get('initial') is not None:
            initial = defaults['initial']
            if callable(initial):
                initial = initial()
            defaults['initial'] = [i.pk for i in initial]
        return super().formfield(**defaults)

    def db_check(self, connection):
        return None

    def db_type(self, connection):
        return None

    def db_parameters(self, connection):
        return {'type': None, 'check': None}