from __future__ import annotations
import copy
import datetime as dt
import decimal
import inspect
import json
import typing
import uuid
import warnings
from abc import ABCMeta
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from functools import lru_cache
from marshmallow import base, class_registry, types
from marshmallow import fields as ma_fields
from marshmallow.decorators import (
from marshmallow.error_store import ErrorStore
from marshmallow.exceptions import StringNotCollectionError, ValidationError
from marshmallow.orderedset import OrderedSet
from marshmallow.utils import (
from marshmallow.warnings import RemovedInMarshmallow4Warning
class SchemaMeta(ABCMeta):
    """Metaclass for the Schema class. Binds the declared fields to
    a ``_declared_fields`` attribute, which is a dictionary mapping attribute
    names to field objects. Also sets the ``opts`` class attribute, which is
    the Schema class's ``class Meta`` options.
    """

    def __new__(mcs, name, bases, attrs):
        meta = attrs.get('Meta')
        ordered = getattr(meta, 'ordered', False)
        if not ordered:
            for base_ in bases:
                if hasattr(base_, 'Meta') and hasattr(base_.Meta, 'ordered'):
                    ordered = base_.Meta.ordered
                    break
            else:
                ordered = False
        cls_fields = _get_fields(attrs)
        for field_name, _ in cls_fields:
            del attrs[field_name]
        klass = super().__new__(mcs, name, bases, attrs)
        inherited_fields = _get_fields_by_mro(klass)
        meta = klass.Meta
        klass.opts = klass.OPTIONS_CLASS(meta, ordered=ordered)
        cls_fields += list(klass.opts.include.items())
        klass._declared_fields = mcs.get_declared_fields(klass=klass, cls_fields=cls_fields, inherited_fields=inherited_fields, dict_cls=dict)
        return klass

    @classmethod
    def get_declared_fields(mcs, klass: type, cls_fields: list, inherited_fields: list, dict_cls: type=dict):
        """Returns a dictionary of field_name => `Field` pairs declared on the class.
        This is exposed mainly so that plugins can add additional fields, e.g. fields
        computed from class Meta options.

        :param klass: The class object.
        :param cls_fields: The fields declared on the class, including those added
            by the ``include`` class Meta option.
        :param inherited_fields: Inherited fields.
        :param dict_cls: dict-like class to use for dict output Default to ``dict``.
        """
        return dict_cls(inherited_fields + cls_fields)

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name and cls.opts.register:
            class_registry.register(name, cls)
        cls._hooks = cls.resolve_hooks()

    def resolve_hooks(cls) -> dict[types.Tag, list[str]]:
        """Add in the decorated processors

        By doing this after constructing the class, we let standard inheritance
        do all the hard work.
        """
        mro = inspect.getmro(cls)
        hooks = defaultdict(list)
        for attr_name in dir(cls):
            for parent in mro:
                try:
                    attr = parent.__dict__[attr_name]
                except KeyError:
                    continue
                else:
                    break
            else:
                continue
            try:
                hook_config = attr.__marshmallow_hook__
            except AttributeError:
                pass
            else:
                for key in hook_config.keys():
                    hooks[key].append(attr_name)
        return hooks