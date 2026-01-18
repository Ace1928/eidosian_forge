import importlib
from django.apps import apps
from django.conf import settings
from django.core.serializers.base import SerializerDoesNotExist
def _load_serializers():
    """
    Register built-in and settings-defined serializers. This is done lazily so
    that user code has a chance to (e.g.) set up custom settings without
    needing to be careful of import order.
    """
    global _serializers
    serializers = {}
    for format in BUILTIN_SERIALIZERS:
        register_serializer(format, BUILTIN_SERIALIZERS[format], serializers)
    if hasattr(settings, 'SERIALIZATION_MODULES'):
        for format in settings.SERIALIZATION_MODULES:
            register_serializer(format, settings.SERIALIZATION_MODULES[format], serializers)
    _serializers = serializers