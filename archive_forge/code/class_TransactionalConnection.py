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
class TransactionalConnection(BaseConnection):
    """A connection specific to one transaction.

  It is possible to pass the transaction and entity group to the
  constructor, but typically the transaction is lazily created by
  _get_transaction() when the first operation is started.
  """
    OPEN = 0
    COMMIT_IN_FLIGHT = 1
    FAILED = 2
    CLOSED = 3

    @_positional(1)
    def __init__(self, adapter=None, config=None, transaction=None, entity_group=None, _api_version=_DATASTORE_V3, previous_transaction=None, mode=TransactionMode.UNKNOWN):
        """Constructor.

    All arguments should be specified as keyword arguments.

    Args:
      adapter: Optional AbstractAdapter subclass instance;
        default IdentityAdapter.
      config: Optional Configuration object.
      transaction: Optional datastore_db.Transaction object.
      entity_group: Deprecated, do not use.
      previous_transaction: Optional datastore_db.Transaction object
        representing the transaction being reset.
      mode: Optional datastore_db.TransactionMode representing the transaction
        mode.

    Raises:
      datastore_errors.BadArgumentError: If previous_transaction and transaction
        are both set.
    """
        super(TransactionalConnection, self).__init__(adapter=adapter, config=config, _api_version=_api_version)
        self._state = TransactionalConnection.OPEN
        if previous_transaction is not None and transaction is not None:
            raise datastore_errors.BadArgumentError('Only one of transaction and previous_transaction should be set')
        self.__adapter = self.adapter
        self.__config = self.config
        if transaction is None:
            app = TransactionOptions.app(self.config)
            app = datastore_types.ResolveAppId(TransactionOptions.app(self.config))
            self.__transaction_rpc = self.async_begin_transaction(None, app, previous_transaction, mode)
        else:
            if self._api_version == _CLOUD_DATASTORE_V1:
                txn_class = six_subset.binary_type
            else:
                txn_class = datastore_pb.Transaction
            if not isinstance(transaction, txn_class):
                raise datastore_errors.BadArgumentError('Invalid transaction (%r)' % transaction)
            self.__transaction = transaction
            self.__transaction_rpc = None
        self.__pending_v1_upserts = {}
        self.__pending_v1_deletes = {}

    @property
    def finished(self):
        return self._state != TransactionalConnection.OPEN

    @property
    def transaction(self):
        """The current transaction. None when state == FINISHED."""
        if self.__transaction_rpc is not None:
            self.__transaction = self.__transaction_rpc.get_result()
            self.__transaction_rpc = None
        return self.__transaction

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

    def async_put(self, config, entities, extra_hook=None):
        """Transactional asynchronous Put operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      entities: An iterable of user-level entity objects.
      extra_hook: Optional function to be called on the result once the
        RPC has completed.

     Returns:
      A MultiRpc object.

    NOTE: If any of the entities has an incomplete key, this will
    *not* patch up those entities with the complete key.
    """
        if self._api_version != _CLOUD_DATASTORE_V1:
            return super(TransactionalConnection, self).async_put(config, entities, extra_hook)
        v1_entities = [self.adapter.entity_to_pb_v1(entity) for entity in entities]
        v1_req = googledatastore.AllocateIdsRequest()
        for v1_entity in v1_entities:
            if not datastore_pbs.is_complete_v1_key(v1_entity.key):
                v1_req.keys.add().CopyFrom(v1_entity.key)
        user_data = (v1_entities, extra_hook)
        service_name = _CLOUD_DATASTORE_V1
        if not v1_req.keys:
            service_name = _NOOP_SERVICE
        return self._make_rpc_call(config, 'AllocateIds', v1_req, googledatastore.AllocateIdsResponse(), get_result_hook=self.__v1_put_allocate_ids_hook, user_data=user_data, service_name=service_name)

    def __v1_put_allocate_ids_hook(self, rpc):
        """Internal method used as get_result_hook for AllocateIds call."""
        self.check_rpc_success(rpc)
        v1_resp = rpc.response
        return self.__v1_build_put_result(list(v1_resp.keys), rpc.user_data)

    def __v1_build_put_result(self, v1_allocated_keys, user_data):
        """Internal method that builds the result of a put operation.

    Converts the results from a v1 AllocateIds operation to a list of user-level
    key objects.

    Args:
      v1_allocated_keys: a list of googledatastore.Keys that have been allocated
      user_data: a tuple consisting of:
        - a list of googledatastore.Entity objects
        - an optional extra_hook
    """
        v1_entities, extra_hook = user_data
        keys = []
        idx = 0
        for v1_entity in v1_entities:
            v1_entity = copy.deepcopy(v1_entity)
            if not datastore_pbs.is_complete_v1_key(v1_entity.key):
                v1_entity.key.CopyFrom(v1_allocated_keys[idx])
                idx += 1
            hashable_key = datastore_types.ReferenceToKeyValue(v1_entity.key)
            self.__pending_v1_deletes.pop(hashable_key, None)
            self.__pending_v1_upserts[hashable_key] = v1_entity
            keys.append(self.adapter.pb_v1_to_key(copy.deepcopy(v1_entity.key)))
        if extra_hook:
            keys = extra_hook(keys)
        return keys

    def async_delete(self, config, keys, extra_hook=None):
        """Transactional asynchronous Delete operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      keys: An iterable of user-level key objects.
      extra_hook: Optional function to be called once the RPC has completed.

    Returns:
      A MultiRpc object.
    """
        if self._api_version != _CLOUD_DATASTORE_V1:
            return super(TransactionalConnection, self).async_delete(config, keys, extra_hook)
        v1_keys = [self.__adapter.key_to_pb_v1(key) for key in keys]
        for key in v1_keys:
            hashable_key = datastore_types.ReferenceToKeyValue(key)
            self.__pending_v1_upserts.pop(hashable_key, None)
            self.__pending_v1_deletes[hashable_key] = key
        return self._make_rpc_call(config, 'Commit', None, googledatastore.CommitResponse(), get_result_hook=self.__v1_delete_hook, user_data=extra_hook, service_name=_NOOP_SERVICE)

    def __v1_delete_hook(self, rpc):
        extra_hook = rpc.user_data
        if extra_hook:
            extra_hook(None)

    def commit(self):
        """Synchronous Commit operation.

    Returns:
      True if the transaction was successfully committed.  False if
      the backend reported a concurrent transaction error.
    """
        rpc = self._create_rpc(service_name=self._api_version)
        rpc = self.async_commit(rpc)
        if rpc is None:
            return True
        return rpc.get_result()

    def async_commit(self, config):
        """Asynchronous Commit operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.

    Returns:
      A MultiRpc object.
    """
        self.wait_for_all_pending_rpcs()
        if self._state != TransactionalConnection.OPEN:
            raise datastore_errors.BadRequestError('Transaction is already finished.')
        self._state = TransactionalConnection.COMMIT_IN_FLIGHT
        transaction = self.transaction
        if transaction is None:
            self._state = TransactionalConnection.CLOSED
            return None
        if self._api_version == _CLOUD_DATASTORE_V1:
            req = googledatastore.CommitRequest()
            req.transaction = transaction
            if Configuration.force_writes(config, self.__config):
                self.__force(req)
            for entity in self.__pending_v1_upserts.values():
                mutation = req.mutations.add()
                mutation.upsert.CopyFrom(entity)
            for key in self.__pending_v1_deletes.values():
                mutation = req.mutations.add()
                mutation.delete.CopyFrom(key)
            self.__pending_v1_upserts.clear()
            self.__pending_v1_deletes.clear()
            resp = googledatastore.CommitResponse()
        else:
            req = transaction
            resp = datastore_pb.CommitResponse()
        return self._make_rpc_call(config, 'Commit', req, resp, get_result_hook=self.__commit_hook, service_name=self._api_version)

    def __commit_hook(self, rpc):
        """Internal method used as get_result_hook for Commit."""
        try:
            rpc.check_success()
            self._state = TransactionalConnection.CLOSED
            self.__transaction = None
        except apiproxy_errors.ApplicationError as err:
            self._state = TransactionalConnection.FAILED
            if err.application_error == datastore_pb.Error.CONCURRENT_TRANSACTION:
                return False
            else:
                raise _ToDatastoreError(err)
        else:
            return True

    def rollback(self):
        """Synchronous Rollback operation."""
        rpc = self.async_rollback(None)
        if rpc is None:
            return None
        return rpc.get_result()

    def async_rollback(self, config):
        """Asynchronous Rollback operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.

     Returns:
      A MultiRpc object.
    """
        self.wait_for_all_pending_rpcs()
        if not (self._state == TransactionalConnection.OPEN or self._state == TransactionalConnection.FAILED):
            raise datastore_errors.BadRequestError('Cannot rollback transaction that is neither OPEN or FAILED state.')
        transaction = self.transaction
        if transaction is None:
            return None
        self._state = TransactionalConnection.CLOSED
        self.__transaction = None
        if self._api_version == _CLOUD_DATASTORE_V1:
            req = googledatastore.RollbackRequest()
            req.transaction = transaction
            resp = googledatastore.RollbackResponse()
        else:
            req = transaction
            resp = api_base_pb.VoidProto()
        return self._make_rpc_call(config, 'Rollback', req, resp, get_result_hook=self.__rollback_hook, service_name=self._api_version)

    def __rollback_hook(self, rpc):
        """Internal method used as get_result_hook for Rollback."""
        self.check_rpc_success(rpc)