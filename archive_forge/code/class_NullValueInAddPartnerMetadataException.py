from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
class NullValueInAddPartnerMetadataException(exceptions.Error):
    """Null values not allowed in add-partner-metadata."""