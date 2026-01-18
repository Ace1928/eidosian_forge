import importlib
from django.apps import apps
from django.conf import settings
from django.core.serializers.base import SerializerDoesNotExist
class BadSerializer:
    """
    Stub serializer to hold exception raised during registration

    This allows the serializer registration to cache serializers and if there
    is an error raised in the process of creating a serializer it will be
    raised and passed along to the caller when the serializer is used.
    """
    internal_use_only = False

    def __init__(self, exception):
        self.exception = exception

    def __call__(self, *args, **kwargs):
        raise self.exception