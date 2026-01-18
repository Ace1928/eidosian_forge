import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
class CreateError(Exception):
    """
    Used internally as a consistent exception type to catch from save (see the
    docstring for SessionBase.save() for details).
    """
    pass