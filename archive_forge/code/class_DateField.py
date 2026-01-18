import datetime
import json
import numbers
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
from apitools.base.py import util
class DateField(six.with_metaclass(_FieldMeta, messages.Field)):
    """Field definition for Date values."""
    VARIANTS = frozenset([messages.Variant.STRING])
    DEFAULT_VARIANT = messages.Variant.STRING
    type = datetime.date