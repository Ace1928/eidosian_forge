import getpass
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS
def _get_pass(self, prompt='Password: '):
    p = getpass.getpass(prompt=prompt)
    if not p:
        raise CommandError('aborted')
    return p