from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import copy
import functools
import os
from googlecloudsdk.api_lib.storage import retry_util as storage_retry_util
from googlecloudsdk.api_lib.storage.gcs_grpc import grpc_util
from googlecloudsdk.api_lib.storage.gcs_grpc import metadata_util
from googlecloudsdk.api_lib.storage.gcs_grpc import retry_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
import six
def _upload_write_object_request_generator(self, first_message):
    """Yields the WriteObjectRequest for each chunk of the source stream.

    The amount_of_data_sent_so_far is equal to the number of bytes read
    from the source stream.

    If _chunk_size is not None, this function will yield the WriteObjectRequest
    object until the amount_of_data_sent_so_far is equal to or greater than the
    value of the new _chunk_size and the length of data sent in the last
    WriteObjectRequest is equal to MAX_WRITE_CHUNK_BYTES, or if there are no
    data in the source stream.

    MAX_WRITE_CHUNK_BYTES is a multiple 256 KiB.

    Clients must only send data that is a multiple of 256 KiB per message,
    unless the object is being finished with``finish_write`` set to ``true``.

    This means that if amount_of_data_sent_so_far >= _chunk_size,
    it must also be ensured before stopping yielding
    requests(WriteObjectRequest) that all requests have sent
    data multiple of 256 KiB, in other words length of data % 256 KiB is 0.

    The source stream data is read in chunks of MAX_WRITE_CHUNK_BYTES, that
    means that each request yielded will send data of size
    MAX_WRITE_CHUNK_BYTES, except if there is a last request before the final
    request(``finish_write`` set to ``true``) where the data length is less
    than MAX_WRITE_CHUNK_BYTES, this means if the length of data in the last
    request yielded is equal to MAX_WRITE_CHUNK_BYTES, all requests sent before
    have sent data of size MAX_WRITE_CHUNK_BYTES, therefore all requests have
    sent data that is multiple of 256 KiB, thus satisfying the condition
    stated before. If the the length of data in the last request yielded is not
    equal to MAX_WRITE_CHUNK_BYTES, then stop when there are no data
    in the source stream(the final request is sent).

    Otherwise if _chunk_size is None, it will yield all WriteObjectRequest
    objects until there are no data in the source stream.

    Args:
      first_message (WriteObjectSpec|str): WriteObjectSpec for Simple uploads,
      str that is the upload id for Resumable and Streaming uploads.

    Yields:
      (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.WriteObjectRequest)
      WriteObjectRequest instance.
    """
    first_request_done = False
    if isinstance(first_message, self._client.types.WriteObjectSpec):
        write_object_spec = first_message
        upload_id = None
    else:
        write_object_spec = None
        upload_id = first_message
    self._initialize_generator()
    while True:
        data = self._source_stream.read(self._client.types.ServiceConstants.Values.MAX_WRITE_CHUNK_BYTES)
        if not first_request_done:
            first_request_done = True
        else:
            write_object_spec = None
            upload_id = None
        if data:
            object_checksums = None
            finish_write = False
        else:
            object_checksums = self._client.types.ObjectChecksums(md5_hash=self._get_md5_hash_if_given())
            finish_write = True
        yield self._client.types.WriteObjectRequest(write_object_spec=write_object_spec, upload_id=upload_id, write_offset=self._uploaded_so_far, checksummed_data=self._client.types.ChecksummedData(content=data), object_checksums=object_checksums, finish_write=finish_write)
        self._uploaded_so_far += len(data)
        if finish_write:
            self._source_stream_finished = True
            break
        if self._chunk_size is None:
            continue
        is_length_of_data_equal_to_max_write_chunk_bytes = len(data) == self._client.types.ServiceConstants.Values.MAX_WRITE_CHUNK_BYTES
        amount_of_data_sent_so_far = self._uploaded_so_far - self._start_offset
        if is_length_of_data_equal_to_max_write_chunk_bytes and amount_of_data_sent_so_far >= self._chunk_size:
            break