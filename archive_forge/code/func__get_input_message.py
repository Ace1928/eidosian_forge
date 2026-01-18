import getpass
import os
import sys
from django.contrib.auth import get_user_model
from django.contrib.auth.management import get_default_username
from django.contrib.auth.password_validation import validate_password
from django.core import exceptions
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS
from django.utils.functional import cached_property
from django.utils.text import capfirst
def _get_input_message(self, field, default=None):
    return '%s%s%s: ' % (capfirst(field.verbose_name), " (leave blank to use '%s')" % default if default else '', ' (%s.%s)' % (field.remote_field.model._meta.object_name, field.m2m_target_field_name() if field.many_to_many else field.remote_field.field_name) if field.remote_field else '')