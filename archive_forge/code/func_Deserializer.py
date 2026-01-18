import json
from django.core.serializers.base import DeserializationError
from django.core.serializers.json import DjangoJSONEncoder
from django.core.serializers.python import Deserializer as PythonDeserializer
from django.core.serializers.python import Serializer as PythonSerializer
def Deserializer(stream_or_string, **options):
    """Deserialize a stream or string of JSON data."""
    if isinstance(stream_or_string, bytes):
        stream_or_string = stream_or_string.decode()
    if isinstance(stream_or_string, (bytes, str)):
        stream_or_string = stream_or_string.split('\n')
    for line in stream_or_string:
        if not line.strip():
            continue
        try:
            yield from PythonDeserializer([json.loads(line)], **options)
        except (GeneratorExit, DeserializationError):
            raise
        except Exception as exc:
            raise DeserializationError() from exc