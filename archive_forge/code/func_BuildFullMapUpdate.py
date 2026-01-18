from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def BuildFullMapUpdate(clear, remove_keys, set_entries, initial_entries, entry_cls, env_builder):
    """Builds the patch environment for an environment update.

  To be used when BuildPartialUpdate cannot be used due to lack of support for
  field masks containing map keys.

  Follows the environments update semantic which applies operations
  in an effective order of clear -> remove -> set.

  Leading and trailing whitespace is stripped from elements in remove_keys
  and the keys of set_entries.

  Args:
    clear: bool, If true, the patch removes existing keys.
    remove_keys: iterable(string), Iterable of keys to remove.
    set_entries: {string: string}, Dict containing entries to set.
    initial_entries: [AdditionalProperty], list of AdditionalProperty class with
      key and value fields, representing starting dict to update from.
    entry_cls: AdditionalProperty, The AdditionalProperty class for the type of
      entry being updated.
    env_builder: [AdditionalProperty] -> Environment, A function which produces
      a patch Environment with the given list of entry_cls properties.

  Returns:
    Environment, a patch environment produced by env_builder.
  """
    entries_dict = collections.OrderedDict(((entry.key, entry.value) for entry in initial_entries))
    if clear:
        entries_dict = collections.OrderedDict()
    remove_keys = set((k.strip() for k in remove_keys or []))
    for key in remove_keys:
        if key in entries_dict:
            del entries_dict[key]
    set_entries = collections.OrderedDict(((k.strip(), v) for k, v in sorted(six.iteritems(set_entries or {}))))
    entries_dict.update(set_entries)
    return env_builder([entry_cls(key=key, value=value) for key, value in six.iteritems(entries_dict)])