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
@Cache
def _SafeLoadYamlFile(path):
    return yaml.safe_load(pkg_resources.GetResourceFromFile(path))