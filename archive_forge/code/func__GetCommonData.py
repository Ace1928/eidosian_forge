from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import importlib
import os
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_release_tracks
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import pkg_resources
from ruamel import yaml
import six
def _GetCommonData(self, attribute_path):
    if not common_data:
        raise LayoutException('Command [{}] references [common command] data but it does not exist.'.format(impl_path))
    return self._GetAttribute(common_data, attribute_path, 'common command')