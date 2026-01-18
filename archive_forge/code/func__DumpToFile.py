from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _DumpToFile(tree, f):
    """Dump helper."""
    from googlecloudsdk.core.resource import resource_printer
    from googlecloudsdk.core.resource import resource_projector
    resource_printer.Print(resource_projector.MakeSerializable(_Serialize(tree)), 'json', out=f)