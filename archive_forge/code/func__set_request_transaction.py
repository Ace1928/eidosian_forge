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
def _set_request_transaction(self, request):
    """Set the current transaction on a request.

    This accesses the transaction property.  The transaction object
    returned is both set as the transaction field on the request
    object and returned.

    Args:
      request: A protobuf with a transaction field.

    Returns:
      An object representing a transaction or None.

    Raises:
      ValueError: if called with a non-Cloud Datastore request when using
          Cloud Datastore.
    """
    if self.finished:
        raise datastore_errors.BadRequestError('Cannot start a new operation in a finished transaction.')
    transaction = self.transaction
    if self._api_version == _CLOUD_DATASTORE_V1:
        if isinstance(request, (googledatastore.CommitRequest, googledatastore.RollbackRequest)):
            request.transaction = transaction
        elif isinstance(request, (googledatastore.LookupRequest, googledatastore.RunQueryRequest)):
            request.read_options.transaction = transaction
        else:
            raise ValueError('Cannot use Cloud Datastore V1 transactions with %s.' % type(request))
        request.read_options.transaction = transaction
    else:
        request.mutable_transaction().CopyFrom(transaction)
    return transaction