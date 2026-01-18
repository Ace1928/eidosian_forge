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
class ManyToManyQuery(ModelSelect):

    def __init__(self, instance, accessor, rel, *args, **kwargs):
        self._instance = instance
        self._accessor = accessor
        self._src_attr = accessor.src_fk.rel_field.name
        self._dest_attr = accessor.dest_fk.rel_field.name
        super(ManyToManyQuery, self).__init__(rel, (rel,), *args, **kwargs)

    def _id_list(self, model_or_id_list):
        if isinstance(model_or_id_list[0], Model):
            return [getattr(obj, self._dest_attr) for obj in model_or_id_list]
        return model_or_id_list

    def add(self, value, clear_existing=False):
        if clear_existing:
            self.clear()
        accessor = self._accessor
        src_id = getattr(self._instance, self._src_attr)
        if isinstance(value, SelectQuery):
            query = value.columns(Value(src_id), accessor.dest_fk.rel_field)
            accessor.through_model.insert_from(fields=[accessor.src_fk, accessor.dest_fk], query=query).execute()
        else:
            value = ensure_tuple(value)
            if not value:
                return
            inserts = [{accessor.src_fk.name: src_id, accessor.dest_fk.name: rel_id} for rel_id in self._id_list(value)]
            accessor.through_model.insert_many(inserts).execute()

    def remove(self, value):
        src_id = getattr(self._instance, self._src_attr)
        if isinstance(value, SelectQuery):
            column = getattr(value.model, self._dest_attr)
            subquery = value.columns(column)
            return self._accessor.through_model.delete().where(self._accessor.dest_fk << subquery & (self._accessor.src_fk == src_id)).execute()
        else:
            value = ensure_tuple(value)
            if not value:
                return
            return self._accessor.through_model.delete().where(self._accessor.dest_fk << self._id_list(value) & (self._accessor.src_fk == src_id)).execute()

    def clear(self):
        src_id = getattr(self._instance, self._src_attr)
        return self._accessor.through_model.delete().where(self._accessor.src_fk == src_id).execute()