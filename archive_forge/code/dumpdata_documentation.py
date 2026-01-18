import gzip
import os
import warnings
from django.apps import apps
from django.core import serializers
from django.core.management.base import BaseCommand, CommandError
from django.core.management.utils import parse_apps_and_model_labels
from django.db import DEFAULT_DB_ALIAS, router

            Collate the objects to be serialized. If count_only is True, just
            count the number of objects to be serialized.
            