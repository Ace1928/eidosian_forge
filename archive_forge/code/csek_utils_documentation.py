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
Search for the unique key corresponding to a given resource.

    Args:
      resource: the resource to find a key for.
      raise_if_missing: bool, raise an exception if the resource is not found.

    Returns: CsekKeyBase, corresponding to the resource, or None if not found
      and not raise_if_missing.

    Raises:
      InvalidKeyFileException: if there are two records matching the resource.
      MissingCsekException: if raise_if_missing and no key is found
        for the provided resource.
    