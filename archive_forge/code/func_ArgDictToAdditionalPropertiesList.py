from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import tempfile
from apitools.base.protorpclite.messages import DecodeError
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import transfer
from googlecloudsdk.api_lib.genomics import exceptions as genomics_exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
import six
def ArgDictToAdditionalPropertiesList(argdict, message):
    result = []
    if argdict is None:
        return result
    for k, v in sorted(six.iteritems(argdict)):
        result.append(message(key=k, value=v))
    return result