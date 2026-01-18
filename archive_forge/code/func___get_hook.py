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
def __get_hook(self, rpc):
    """Internal method used as get_result_hook for Get operation."""
    self.check_rpc_success(rpc)
    config, keys_from_request, extra_hook = rpc.user_data
    if self._api_version == _DATASTORE_V3 and rpc.response.in_order():
        entities = []
        for entity_result in rpc.response.entity_list():
            if entity_result.has_entity():
                entity = self.__adapter.pb_to_entity(entity_result.entity())
            else:
                entity = None
            entities.append(entity)
    else:
        current_get_response = rpc.response
        result_dict = {}
        self.__add_get_response_entities_to_dict(current_get_response, result_dict)
        deferred_req = copy.deepcopy(rpc.request)
        if self._api_version == _CLOUD_DATASTORE_V1:
            method = 'Lookup'
            deferred_resp = googledatastore.LookupResponse()
            while current_get_response.deferred:
                deferred_req.ClearField('keys')
                deferred_req.keys.extend(current_get_response.deferred)
                deferred_resp.Clear()
                deferred_rpc = self._make_rpc_call(config, method, deferred_req, deferred_resp, service_name=self._api_version)
                deferred_rpc.get_result()
                current_get_response = deferred_rpc.response
                self.__add_get_response_entities_to_dict(current_get_response, result_dict)
        else:
            method = 'Get'
            deferred_resp = datastore_pb.GetResponse()
            while current_get_response.deferred_list():
                deferred_req.clear_key()
                deferred_req.key_list().extend(current_get_response.deferred_list())
                deferred_resp.Clear()
                deferred_rpc = self._make_rpc_call(config, method, deferred_req, deferred_resp, service_name=self._api_version)
                deferred_rpc.get_result()
                current_get_response = deferred_rpc.response
                self.__add_get_response_entities_to_dict(current_get_response, result_dict)
        entities = [result_dict.get(datastore_types.ReferenceToKeyValue(pb)) for pb in keys_from_request]
    if extra_hook is not None:
        entities = extra_hook(entities)
    return entities