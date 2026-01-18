import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import FieldError
from django.db import (
from django.db.models import Q, Window, signals
from django.db.models.functions import RowNumber
from django.db.models.lookups import GreaterThan, LessThanOrEqual
from django.db.models.query import QuerySet
from django.db.models.query_utils import DeferredAttribute
from django.db.models.utils import AltersData, resolve_callables
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
class ReverseOneToOneDescriptor:
    """
    Accessor to the related object on the reverse side of a one-to-one
    relation.

    In the example::

        class Restaurant(Model):
            place = OneToOneField(Place, related_name='restaurant')

    ``Place.restaurant`` is a ``ReverseOneToOneDescriptor`` instance.
    """

    def __init__(self, related):
        self.related = related

    @cached_property
    def RelatedObjectDoesNotExist(self):
        return type('RelatedObjectDoesNotExist', (self.related.related_model.DoesNotExist, AttributeError), {'__module__': self.related.model.__module__, '__qualname__': '%s.%s.RelatedObjectDoesNotExist' % (self.related.model.__qualname__, self.related.name)})

    def is_cached(self, instance):
        return self.related.is_cached(instance)

    def get_queryset(self, **hints):
        return self.related.related_model._base_manager.db_manager(hints=hints).all()

    def get_prefetch_queryset(self, instances, queryset=None):
        warnings.warn('get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() instead.', RemovedInDjango60Warning, stacklevel=2)
        if queryset is None:
            return self.get_prefetch_querysets(instances)
        return self.get_prefetch_querysets(instances, [queryset])

    def get_prefetch_querysets(self, instances, querysets=None):
        if querysets and len(querysets) != 1:
            raise ValueError('querysets argument of get_prefetch_querysets() should have a length of 1.')
        queryset = querysets[0] if querysets else self.get_queryset()
        queryset._add_hints(instance=instances[0])
        rel_obj_attr = self.related.field.get_local_related_value
        instance_attr = self.related.field.get_foreign_related_value
        instances_dict = {instance_attr(inst): inst for inst in instances}
        query = {'%s__in' % self.related.field.name: instances}
        queryset = queryset.filter(**query)
        for rel_obj in queryset:
            instance = instances_dict[rel_obj_attr(rel_obj)]
            self.related.field.set_cached_value(rel_obj, instance)
        return (queryset, rel_obj_attr, instance_attr, True, self.related.get_cache_name(), False)

    def __get__(self, instance, cls=None):
        """
        Get the related instance through the reverse relation.

        With the example above, when getting ``place.restaurant``:

        - ``self`` is the descriptor managing the ``restaurant`` attribute
        - ``instance`` is the ``place`` instance
        - ``cls`` is the ``Place`` class (unused)

        Keep in mind that ``Restaurant`` holds the foreign key to ``Place``.
        """
        if instance is None:
            return self
        try:
            rel_obj = self.related.get_cached_value(instance)
        except KeyError:
            related_pk = instance.pk
            if related_pk is None:
                rel_obj = None
            else:
                filter_args = self.related.field.get_forward_related_filter(instance)
                try:
                    rel_obj = self.get_queryset(instance=instance).get(**filter_args)
                except self.related.related_model.DoesNotExist:
                    rel_obj = None
                else:
                    self.related.field.set_cached_value(rel_obj, instance)
            self.related.set_cached_value(instance, rel_obj)
        if rel_obj is None:
            raise self.RelatedObjectDoesNotExist('%s has no %s.' % (instance.__class__.__name__, self.related.get_accessor_name()))
        else:
            return rel_obj

    def __set__(self, instance, value):
        """
        Set the related instance through the reverse relation.

        With the example above, when setting ``place.restaurant = restaurant``:

        - ``self`` is the descriptor managing the ``restaurant`` attribute
        - ``instance`` is the ``place`` instance
        - ``value`` is the ``restaurant`` instance on the right of the equal sign

        Keep in mind that ``Restaurant`` holds the foreign key to ``Place``.
        """
        if value is None:
            rel_obj = self.related.get_cached_value(instance, default=None)
            if rel_obj is not None:
                self.related.delete_cached_value(instance)
                setattr(rel_obj, self.related.field.name, None)
        elif not isinstance(value, self.related.related_model):
            raise ValueError('Cannot assign "%r": "%s.%s" must be a "%s" instance.' % (value, instance._meta.object_name, self.related.get_accessor_name(), self.related.related_model._meta.object_name))
        else:
            if instance._state.db is None:
                instance._state.db = router.db_for_write(instance.__class__, instance=value)
            if value._state.db is None:
                value._state.db = router.db_for_write(value.__class__, instance=instance)
            if not router.allow_relation(value, instance):
                raise ValueError('Cannot assign "%r": the current database router prevents this relation.' % value)
            related_pk = tuple((getattr(instance, field.attname) for field in self.related.field.foreign_related_fields))
            for index, field in enumerate(self.related.field.local_related_fields):
                setattr(value, field.attname, related_pk[index])
            self.related.set_cached_value(instance, value)
            self.related.field.set_cached_value(value, instance)

    def __reduce__(self):
        return (getattr, (self.related.model, self.related.name))