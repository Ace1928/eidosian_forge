from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
import six
class MultipleAppYamlError(Exception):
    """An application configuration has more than one valid app yaml files."""