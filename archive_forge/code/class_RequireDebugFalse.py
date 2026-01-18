import logging
import logging.config  # needed when logging_config doesn't start with logging.config
from copy import copy
from django.conf import settings
from django.core import mail
from django.core.mail import get_connection
from django.core.management.color import color_style
from django.utils.module_loading import import_string
class RequireDebugFalse(logging.Filter):

    def filter(self, record):
        return not settings.DEBUG