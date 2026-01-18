from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import copy
import json
import os
from googlecloudsdk.command_lib.anthos import flags
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
from six.moves import urllib
def _ParseTags(self, tags):
    return ','.join(['{}={}'.format(x, y) for x, y in six.iteritems(tags)])