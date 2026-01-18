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
def _ConstructSequenceHelper(self, macro, source_func, data):
    new_list = []
    for i in data:
        if isinstance(i, six.string_types) and i.startswith(macro):
            attribute_path = i[len(macro):]
            for path in attribute_path.split(','):
                new_list.extend(source_func(path))
        else:
            new_list.append(i)
    return new_list