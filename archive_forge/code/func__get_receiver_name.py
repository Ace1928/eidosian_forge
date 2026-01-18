from __future__ import absolute_import
from django.dispatch import Signal
from sentry_sdk import Hub
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.integrations.django import DJANGO_VERSION
def _get_receiver_name(receiver):
    name = ''
    if hasattr(receiver, '__qualname__'):
        name = receiver.__qualname__
    elif hasattr(receiver, '__name__'):
        name = receiver.__name__
    elif hasattr(receiver, 'func'):
        if hasattr(receiver, 'func') and hasattr(receiver.func, '__name__'):
            name = 'partial(<function ' + receiver.func.__name__ + '>)'
    if name == '':
        return str(receiver)
    if hasattr(receiver, '__module__'):
        name = receiver.__module__ + '.' + name
    return name