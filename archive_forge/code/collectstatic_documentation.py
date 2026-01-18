import os
from django.apps import apps
from django.contrib.staticfiles.finders import get_finders
from django.contrib.staticfiles.storage import staticfiles_storage
from django.core.checks import Tags
from django.core.files.storage import FileSystemStorage
from django.core.management.base import BaseCommand, CommandError
from django.core.management.color import no_style
from django.utils.functional import cached_property

        Attempt to copy ``path`` with storage
        