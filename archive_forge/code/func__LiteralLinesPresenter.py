from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _LiteralLinesPresenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')