from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def _CheckForUnusedFields(obj):
    """Check for any unused fields in nested messages or lists."""
    if isinstance(obj, proto_messages.Message):
        unused_fields = obj.all_unrecognized_fields()
        if unused_fields:
            if len(unused_fields) > 1:
                unused_msg = '{%s}' % ','.join(sorted(unused_fields))
            else:
                unused_msg = unused_fields[0]
            raise ValueError('.%s: unused' % unused_msg)
        for used_field in obj.all_fields():
            try:
                field = getattr(obj, used_field.name)
                _CheckForUnusedFields(field)
            except ValueError as e:
                raise ValueError('.%s%s' % (used_field.name, e))
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            try:
                _CheckForUnusedFields(item)
            except ValueError as e:
                raise ValueError('[%d]%s' % (i, e))