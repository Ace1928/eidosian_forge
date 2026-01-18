import functools
import glob
import gzip
import os
import sys
import warnings
import zipfile
from itertools import product
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.core.exceptions import ImproperlyConfigured
from django.core.management.base import BaseCommand, CommandError
from django.core.management.color import no_style
from django.core.management.utils import parse_apps_and_model_labels
from django.db import (
from django.utils.functional import cached_property
class SingleZipReader(zipfile.ZipFile):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.namelist()) != 1:
            raise ValueError('Zip-compressed fixtures must contain one file.')

    def read(self):
        return zipfile.ZipFile.read(self, self.namelist()[0])