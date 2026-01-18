from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import io
import json
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_diff
from googlecloudsdk.core.util import edit
import six
def CheckOnlyAsciiCharactersInAuthorizedView(authorized_view):
    """Raises a ValueError if the view contains non-ascii characters."""
    if authorized_view is None or authorized_view.subsetView is None:
        return
    subset_view = authorized_view.subsetView
    if subset_view.rowPrefixes is not None:
        for row_prefix in subset_view.rowPrefixes:
            CheckAscii(row_prefix)
    if subset_view.familySubsets is not None:
        for additional_property in subset_view.familySubsets.additionalProperties:
            family_subset = additional_property.value
            for qualifier in family_subset.qualifiers:
                CheckAscii(qualifier)
            for qualifier_prefix in family_subset.qualifierPrefixes:
                CheckAscii(qualifier_prefix)