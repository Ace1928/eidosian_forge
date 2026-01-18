import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
class UnsupportedMediaTypeError(WADLError):
    """A media type was given that's not supported in this context.

    A resource can only be bound to media types it has representations
    of.
    """