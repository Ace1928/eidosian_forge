from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def _set_request_read_policy(self, request, config=None):
    """Set the read policy on a request.

    This takes the read policy from the config argument or the
    configuration's default configuration, and sets the request's read
    options.

    Args:
      request: A read request protobuf.
      config: Optional Configuration object.

    Returns:
      True if the read policy specifies a read current request, False if it
        specifies an eventually consistent request, None if it does
        not specify a read consistency.
    """
    if isinstance(config, apiproxy_stub_map.UserRPC):
        read_policy = getattr(config, 'read_policy', None)
    else:
        read_policy = Configuration.read_policy(config)
    if read_policy is None:
        read_policy = self.__config.read_policy
    if hasattr(request, 'set_failover_ms') and hasattr(request, 'strong'):
        if read_policy == Configuration.APPLY_ALL_JOBS_CONSISTENCY:
            request.set_strong(True)
            return True
        elif read_policy == Configuration.EVENTUAL_CONSISTENCY:
            request.set_strong(False)
            request.set_failover_ms(-1)
            return False
        else:
            return None
    elif hasattr(request, 'read_options'):
        if read_policy == Configuration.EVENTUAL_CONSISTENCY:
            request.read_options.read_consistency = googledatastore.ReadOptions.EVENTUAL
            return False
        else:
            return None
    else:
        raise datastore_errors.BadRequestError('read_policy is only supported on read operations.')