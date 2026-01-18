from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _GetAllowlistedCommandVocabulary():
    """Returns allowlisted set of gcloud commands."""
    vocabulary_file = os.path.join(os.path.dirname(__file__), 'gcloud_command_vocabulary.txt')
    return set((line for line in files.ReadFileContents(vocabulary_file).split('\n') if not line.startswith('#')))