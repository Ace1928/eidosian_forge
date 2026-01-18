from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import json
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def MaybeLookupKeyMessage(csek_keys_or_none, resource, compute_client):
    maybe_key = MaybeLookupKey(csek_keys_or_none, resource)
    return MaybeToMessage(maybe_key, compute_client)