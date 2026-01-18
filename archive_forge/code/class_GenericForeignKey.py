import functools
import itertools
import warnings
from collections import defaultdict
from asgiref.sync import sync_to_async
from django.contrib.contenttypes.models import ContentType
from django.core import checks
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import DEFAULT_DB_ALIAS, models, router, transaction
from django.db.models import DO_NOTHING, ForeignObject, ForeignObjectRel
from django.db.models.base import ModelBase, make_foreign_order_accessors
from django.db.models.fields.mixins import FieldCacheMixin
from django.db.models.fields.related import (
from django.db.models.query_utils import PathInfo
from django.db.models.sql import AND
from django.db.models.sql.where import WhereNode
from django.db.models.utils import AltersData
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
class GenericForeignKey(FieldCacheMixin):
    """
    Provide a generic many-to-one relation through the ``content_type`` and
    ``object_id`` fields.

    This class also doubles as an accessor to the related object (similar to
    ForwardManyToOneDescriptor) by adding itself as a model attribute.
    """
    auto_created = False
    concrete = False
    editable = False
    hidden = False
    is_relation = True
    many_to_many = False
    many_to_one = True
    one_to_many = False
    one_to_one = False
    related_model = None
    remote_field = None

    def __init__(self, ct_field='content_type', fk_field='object_id', for_concrete_model=True):
        self.ct_field = ct_field
        self.fk_field = fk_field
        self.for_concrete_model = for_concrete_model
        self.editable = False
        self.rel = None
        self.column = None

    def contribute_to_class(self, cls, name, **kwargs):
        self.name = name
        self.model = cls
        cls._meta.add_field(self, private=True)
        setattr(cls, name, self)

    def get_filter_kwargs_for_object(self, obj):
        """See corresponding method on Field"""
        return {self.fk_field: getattr(obj, self.fk_field), self.ct_field: getattr(obj, self.ct_field)}

    def get_forward_related_filter(self, obj):
        """See corresponding method on RelatedField"""
        return {self.fk_field: obj.pk, self.ct_field: ContentType.objects.get_for_model(obj).pk}

    def __str__(self):
        model = self.model
        return '%s.%s' % (model._meta.label, self.name)

    def check(self, **kwargs):
        return [*self._check_field_name(), *self._check_object_id_field(), *self._check_content_type_field()]

    def _check_field_name(self):
        if self.name.endswith('_'):
            return [checks.Error('Field names must not end with an underscore.', obj=self, id='fields.E001')]
        else:
            return []

    def _check_object_id_field(self):
        try:
            self.model._meta.get_field(self.fk_field)
        except FieldDoesNotExist:
            return [checks.Error("The GenericForeignKey object ID references the nonexistent field '%s'." % self.fk_field, obj=self, id='contenttypes.E001')]
        else:
            return []

    def _check_content_type_field(self):
        """
        Check if field named `field_name` in model `model` exists and is a
        valid content_type field (is a ForeignKey to ContentType).
        """
        try:
            field = self.model._meta.get_field(self.ct_field)
        except FieldDoesNotExist:
            return [checks.Error("The GenericForeignKey content type references the nonexistent field '%s.%s'." % (self.model._meta.object_name, self.ct_field), obj=self, id='contenttypes.E002')]
        else:
            if not isinstance(field, models.ForeignKey):
                return [checks.Error("'%s.%s' is not a ForeignKey." % (self.model._meta.object_name, self.ct_field), hint="GenericForeignKeys must use a ForeignKey to 'contenttypes.ContentType' as the 'content_type' field.", obj=self, id='contenttypes.E003')]
            elif field.remote_field.model != ContentType:
                return [checks.Error("'%s.%s' is not a ForeignKey to 'contenttypes.ContentType'." % (self.model._meta.object_name, self.ct_field), hint="GenericForeignKeys must use a ForeignKey to 'contenttypes.ContentType' as the 'content_type' field.", obj=self, id='contenttypes.E004')]
            else:
                return []

    def get_cache_name(self):
        return self.name

    def get_content_type(self, obj=None, id=None, using=None, model=None):
        if obj is not None:
            return ContentType.objects.db_manager(obj._state.db).get_for_model(obj, for_concrete_model=self.for_concrete_model)
        elif id is not None:
            return ContentType.objects.db_manager(using).get_for_id(id)
        elif model is not None:
            return ContentType.objects.db_manager(using).get_for_model(model, for_concrete_model=self.for_concrete_model)
        else:
            raise Exception('Impossible arguments to GFK.get_content_type!')

    def get_prefetch_queryset(self, instances, queryset=None):
        warnings.warn('get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() instead.', RemovedInDjango60Warning, stacklevel=2)
        if queryset is None:
            return self.get_prefetch_querysets(instances)
        return self.get_prefetch_querysets(instances, [queryset])

    def get_prefetch_querysets(self, instances, querysets=None):
        custom_queryset_dict = {}
        if querysets is not None:
            for queryset in querysets:
                ct_id = self.get_content_type(model=queryset.query.model, using=queryset.db).pk
                if ct_id in custom_queryset_dict:
                    raise ValueError('Only one queryset is allowed for each content type.')
                custom_queryset_dict[ct_id] = queryset
        fk_dict = defaultdict(set)
        instance_dict = {}
        ct_attname = self.model._meta.get_field(self.ct_field).get_attname()
        for instance in instances:
            ct_id = getattr(instance, ct_attname)
            if ct_id is not None:
                fk_val = getattr(instance, self.fk_field)
                if fk_val is not None:
                    fk_dict[ct_id].add(fk_val)
                    instance_dict[ct_id] = instance
        ret_val = []
        for ct_id, fkeys in fk_dict.items():
            if ct_id in custom_queryset_dict:
                ret_val.extend(custom_queryset_dict[ct_id].filter(pk__in=fkeys))
            else:
                instance = instance_dict[ct_id]
                ct = self.get_content_type(id=ct_id, using=instance._state.db)
                ret_val.extend(ct.get_all_objects_for_this_type(pk__in=fkeys))

        def gfk_key(obj):
            ct_id = getattr(obj, ct_attname)
            if ct_id is None:
                return None
            else:
                model = self.get_content_type(id=ct_id, using=obj._state.db).model_class()
                return (model._meta.pk.get_prep_value(getattr(obj, self.fk_field)), model)
        return (ret_val, lambda obj: (obj.pk, obj.__class__), gfk_key, True, self.name, False)

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        f = self.model._meta.get_field(self.ct_field)
        ct_id = getattr(instance, f.get_attname(), None)
        pk_val = getattr(instance, self.fk_field)
        rel_obj = self.get_cached_value(instance, default=None)
        if rel_obj is None and self.is_cached(instance):
            return rel_obj
        if rel_obj is not None:
            ct_match = ct_id == self.get_content_type(obj=rel_obj, using=instance._state.db).id
            pk_match = ct_match and rel_obj._meta.pk.to_python(pk_val) == rel_obj.pk
            if pk_match:
                return rel_obj
            else:
                rel_obj = None
        if ct_id is not None:
            ct = self.get_content_type(id=ct_id, using=instance._state.db)
            try:
                rel_obj = ct.get_object_for_this_type(pk=pk_val)
            except ObjectDoesNotExist:
                pass
        self.set_cached_value(instance, rel_obj)
        return rel_obj

    def __set__(self, instance, value):
        ct = None
        fk = None
        if value is not None:
            ct = self.get_content_type(obj=value)
            fk = value.pk
        setattr(instance, self.ct_field, ct)
        setattr(instance, self.fk_field, fk)
        self.set_cached_value(instance, value)