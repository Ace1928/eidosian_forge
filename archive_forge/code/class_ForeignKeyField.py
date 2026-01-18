from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class ForeignKeyField(Field):
    accessor_class = ForeignKeyAccessor
    backref_accessor_class = BackrefAccessor

    def __init__(self, model, field=None, backref=None, on_delete=None, on_update=None, deferrable=None, _deferred=None, rel_model=None, to_field=None, object_id_name=None, lazy_load=True, constraint_name=None, related_name=None, *args, **kwargs):
        kwargs.setdefault('index', True)
        super(ForeignKeyField, self).__init__(*args, **kwargs)
        if rel_model is not None:
            __deprecated__('"rel_model" has been deprecated in favor of "model" for ForeignKeyField objects.')
            model = rel_model
        if to_field is not None:
            __deprecated__('"to_field" has been deprecated in favor of "field" for ForeignKeyField objects.')
            field = to_field
        if related_name is not None:
            __deprecated__('"related_name" has been deprecated in favor of "backref" for Field objects.')
            backref = related_name
        self._is_self_reference = model == 'self'
        self.rel_model = model
        self.rel_field = field
        self.declared_backref = backref
        self.backref = None
        self.on_delete = on_delete
        self.on_update = on_update
        self.deferrable = deferrable
        self.deferred = _deferred
        self.object_id_name = object_id_name
        self.lazy_load = lazy_load
        self.constraint_name = constraint_name

    @property
    def field_type(self):
        if not isinstance(self.rel_field, AutoField):
            return self.rel_field.field_type
        elif isinstance(self.rel_field, BigAutoField):
            return BigIntegerField.field_type
        return IntegerField.field_type

    def get_modifiers(self):
        if not isinstance(self.rel_field, AutoField):
            return self.rel_field.get_modifiers()
        return super(ForeignKeyField, self).get_modifiers()

    def adapt(self, value):
        return self.rel_field.adapt(value)

    def db_value(self, value):
        if isinstance(value, self.rel_model):
            value = getattr(value, self.rel_field.name)
        return self.rel_field.db_value(value)

    def python_value(self, value):
        if isinstance(value, self.rel_model):
            return value
        return self.rel_field.python_value(value)

    def bind(self, model, name, set_attribute=True):
        if not self.column_name:
            self.column_name = name if name.endswith('_id') else name + '_id'
        if not self.object_id_name:
            self.object_id_name = self.column_name
            if self.object_id_name == name:
                self.object_id_name += '_id'
        elif self.object_id_name == name:
            raise ValueError('ForeignKeyField "%s"."%s" specifies an object_id_name that conflicts with its field name.' % (model._meta.name, name))
        if self._is_self_reference:
            self.rel_model = model
        if isinstance(self.rel_field, basestring):
            self.rel_field = getattr(self.rel_model, self.rel_field)
        elif self.rel_field is None:
            self.rel_field = self.rel_model._meta.primary_key
        super(ForeignKeyField, self).bind(model, name, set_attribute)
        self.safe_name = self.object_id_name
        if callable_(self.declared_backref):
            self.backref = self.declared_backref(self)
        else:
            self.backref, self.declared_backref = (self.declared_backref, None)
        if not self.backref:
            self.backref = '%s_set' % model._meta.name
        if set_attribute:
            setattr(model, self.object_id_name, ObjectIdAccessor(self))
            if self.backref not in '!+':
                setattr(self.rel_model, self.backref, self.backref_accessor_class(self))

    def foreign_key_constraint(self):
        parts = []
        if self.constraint_name:
            parts.extend((SQL('CONSTRAINT'), Entity(self.constraint_name)))
        parts.extend([SQL('FOREIGN KEY'), EnclosedNodeList((self,)), SQL('REFERENCES'), self.rel_model, EnclosedNodeList((self.rel_field,))])
        if self.on_delete:
            parts.append(SQL('ON DELETE %s' % self.on_delete))
        if self.on_update:
            parts.append(SQL('ON UPDATE %s' % self.on_update))
        if self.deferrable:
            parts.append(SQL('DEFERRABLE %s' % self.deferrable))
        return NodeList(parts)

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError('Cannot look-up non-existant "__" methods.')
        if attr in self.rel_model._meta.fields:
            return self.rel_model._meta.fields[attr]
        raise AttributeError('Foreign-key has no attribute %s, nor is it a valid field on the related model.' % attr)