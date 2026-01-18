from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
def _process_results(self, results):
    in_memory_filter = self._batch_shared.augmented_query._in_memory_filter
    if in_memory_filter:
        results = list(filter(in_memory_filter, results))
    in_memory_results = self._batch_shared.augmented_query._in_memory_results
    if in_memory_results and self.__next_index < len(in_memory_results):
        original_query = super(_AugmentedBatch, self).query
        if original_query._order:
            if results:
                next_result = in_memory_results[self.__next_index]
                next_key = original_query._order.key(next_result)
                i = 0
                while i < len(results):
                    result = results[i]
                    result_key = original_query._order.key(result)
                    while next_key <= result_key:
                        results.insert(i, next_result)
                        i += 1
                        self.__next_index += 1
                        if self.__next_index >= len(in_memory_results):
                            break
                        next_result = in_memory_results[self.__next_index]
                        next_key = original_query._order.key(next_result)
                    i += 1
        elif results or not super(_AugmentedBatch, self).more_results:
            results = in_memory_results + results
            self.__next_index = len(in_memory_results)
    if self.__in_memory_offset:
        assert not self._skipped_results
        offset = min(self.__in_memory_offset, len(results))
        if offset:
            self._skipped_results += offset
            self.__in_memory_offset -= offset
            results = results[offset:]
    if self.__in_memory_limit is not None:
        results = results[:self.__in_memory_limit]
        self.__in_memory_limit -= len(results)
        if self.__in_memory_limit <= 0:
            self._end()
    return super(_AugmentedBatch, self)._process_results(results)