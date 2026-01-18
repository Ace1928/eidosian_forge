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
def _ParseInitArgs(self, local_dir, description=None, name=None, tags=None, info_url=None, **kwargs):
    del kwargs
    package_path = local_dir
    if not package_path.endswith('/'):
        package_path += '/'
    exec_args = ['init', package_path]
    if description:
        exec_args.extend(['--description', description])
    if name:
        exec_args.extend(['--name', name])
    if tags:
        exec_args.extend(['--tag', self._ParseTags(tags)])
    if info_url:
        exec_args.extend(['--url', info_url])
    return exec_args