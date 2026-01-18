import os
import sys
from io import StringIO
from django.apps import apps
from django.conf import settings
from django.core import serializers
from django.db import router
from django.db.transaction import atomic
from django.utils.module_loading import import_string
def _get_database_display_str(self, verbosity, database_name):
    """
        Return display string for a database for use in various actions.
        """
    return "'%s'%s" % (self.connection.alias, " ('%s')" % database_name if verbosity >= 2 else '')