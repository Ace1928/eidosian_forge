from io import StringIO
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
class SerializerDoesNotExist(KeyError):
    """The requested serializer was not found."""
    pass