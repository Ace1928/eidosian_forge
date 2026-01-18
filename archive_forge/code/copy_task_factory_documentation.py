from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.tasks.cp import copy_managed_folder_task
from googlecloudsdk.command_lib.storage.tasks.cp import daisy_chain_copy_task
from googlecloudsdk.command_lib.storage.tasks.cp import file_download_task
from googlecloudsdk.command_lib.storage.tasks.cp import file_upload_task
from googlecloudsdk.command_lib.storage.tasks.cp import intra_cloud_copy_task
from googlecloudsdk.command_lib.storage.tasks.cp import parallel_composite_upload_util
from googlecloudsdk.command_lib.storage.tasks.cp import streaming_download_task
from googlecloudsdk.command_lib.storage.tasks.cp import streaming_upload_task
Factory method that returns the correct copy task for the arguments.

  Args:
    source_resource (resource_reference.Resource): Reference to file to copy.
    destination_resource (resource_reference.Resource): Reference to destination
      to copy file to.
    delete_source (bool): If copy completes successfully, delete the source
      object afterwards.
    do_not_decompress (bool): Prevents automatically decompressing downloaded
      gzips.
    fetch_source_fields_scope (FieldsScope|None): If present, refetch
      source_resource, populated with metadata determined by this FieldsScope.
      Useful for lazy or parallelized GET calls. Currently only implemented for
      intra-cloud copies.
    force_daisy_chain (bool): If True, yields daisy chain copy tasks in place of
      intra-cloud copy tasks.
    posix_to_set (PosixAttributes|None): Triggers setting POSIX on result of
      copy and avoids re-parsing POSIX info.
    print_created_message (bool): Print the versioned URL of each successfully
      copied object.
    print_source_version (bool): Print source object version in status message
      enabled by the `verbose` kwarg.
    shared_stream (stream): Multiple tasks may reuse this read or write stream.
    user_request_args (UserRequestArgs|None): Values for RequestConfig.
    verbose (bool): Print a "copying" status message on task initialization.

  Returns:
    Task object that can be executed to perform a copy.

  Raises:
    NotImplementedError: Cross-cloud copy.
    Error: Local filesystem copy.
  