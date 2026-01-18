from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import subprocess
import time
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
class DatabaseInstancePresentation(object):
    """Represents a DatabaseInstance message that is modified for user visibility."""

    def __init__(self, orig):
        for field in orig.all_fields():
            if field.name == 'state':
                if orig.settings and orig.settings.activationPolicy == messages.Settings.ActivationPolicyValueValuesEnum.NEVER:
                    self.state = 'STOPPED'
                else:
                    self.state = orig.state
            else:
                value = getattr(orig, field.name)
                if value is not None and (not (isinstance(value, list) and (not value))):
                    if field.name in ['currentDiskSize', 'maxDiskSize']:
                        setattr(self, field.name, six.text_type(value))
                    else:
                        setattr(self, field.name, value)

    def __eq__(self, other):
        """Overrides the default implementation by checking attribute dicts."""
        if isinstance(other, DatabaseInstancePresentation):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """Overrides the default implementation (only needed for Python 2)."""
        return not self.__eq__(other)