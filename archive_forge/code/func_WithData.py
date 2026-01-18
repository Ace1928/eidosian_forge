from io import StringIO
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
@classmethod
def WithData(cls, original_exc, model, fk, field_value):
    """
        Factory method for creating a deserialization error which has a more
        explanatory message.
        """
    return cls("%s: (%s:pk=%s) field_value was '%s'" % (original_exc, model, fk, field_value))