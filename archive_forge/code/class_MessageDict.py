from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
class MessageDict(collections.OrderedDict):
    """A specialized dict that is used for JSON message payloads - Request.arguments,
    Response.body, and Event.body.

    For all members that normally throw KeyError when a requested key is missing, this
    dict raises InvalidMessageError instead. Thus, a message handler can skip checks
    for missing properties, and just work directly with the payload on the assumption
    that it is valid according to the protocol specification; if anything is missing,
    it will be reported automatically in the proper manner.

    If the value for the requested key is itself a dict, it is returned as is, and not
    automatically converted to MessageDict. Thus, to enable convenient chaining - e.g.
    d["a"]["b"]["c"] - the dict must consistently use MessageDict instances rather than
    vanilla dicts for all its values, recursively. This is guaranteed for the payload
    of all freshly received messages (unless and until it is mutated), but there is no
    such guarantee for outgoing messages.
    """

    def __init__(self, message, items=None):
        assert message is None or isinstance(message, Message)
        if items is None:
            super().__init__()
        else:
            super().__init__(items)
        self.message = message
        'The Message object that owns this dict.\n\n        For any instance exposed via a Message object corresponding to some incoming\n        message, it is guaranteed to reference that Message object. There is no similar\n        guarantee for outgoing messages.\n        '

    def __repr__(self):
        try:
            return format(json.repr(self))
        except Exception:
            return super().__repr__()

    def __call__(self, key, validate, optional=False):
        """Like get(), but with validation.

        The item is first retrieved as if with self.get(key, default=()) - the default
        value is () rather than None, so that JSON nulls are distinguishable from
        missing properties.

        If optional=True, and the value is (), it's returned as is. Otherwise, the
        item is validated by invoking validate(item) on it.

        If validate=False, it's treated as if it were (lambda x: x) - i.e. any value
        is considered valid, and is returned unchanged. If validate is a type or a
        tuple, it's treated as json.of_type(validate). Otherwise, if validate is not
        callable(), it's treated as json.default(validate).

        If validate() returns successfully, the item is substituted with the value
        it returns - thus, the validator can e.g. replace () with a suitable default
        value for the property.

        If validate() raises TypeError or ValueError, raises InvalidMessageError with
        the same text that applies_to(self.messages).

        See debugpy.common.json for reusable validators.
        """
        if not validate:
            validate = lambda x: x
        elif isinstance(validate, type) or isinstance(validate, tuple):
            validate = json.of_type(validate, optional=optional)
        elif not callable(validate):
            validate = json.default(validate)
        value = self.get(key, ())
        try:
            value = validate(value)
        except (TypeError, ValueError) as exc:
            message = Message if self.message is None else self.message
            err = str(exc)
            if not err.startswith('['):
                err = ' ' + err
            raise message.isnt_valid('{0}{1}', json.repr(key), err)
        return value

    def _invalid_if_no_key(func):

        def wrap(self, key, *args, **kwargs):
            try:
                return func(self, key, *args, **kwargs)
            except KeyError:
                message = Message if self.message is None else self.message
                raise message.isnt_valid('missing property {0!r}', key)
        return wrap
    __getitem__ = _invalid_if_no_key(collections.OrderedDict.__getitem__)
    __delitem__ = _invalid_if_no_key(collections.OrderedDict.__delitem__)
    pop = _invalid_if_no_key(collections.OrderedDict.pop)
    del _invalid_if_no_key