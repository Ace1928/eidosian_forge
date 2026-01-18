from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import shutil
import sys
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def PrependEnv(name, values):
    paths = GetEnv(name).split(';')
    for value in values:
        if value in paths:
            Remove(paths, value)
        paths.insert(0, value)
    SetEnv(name, ';'.join(paths))