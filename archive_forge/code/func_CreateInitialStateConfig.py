from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def CreateInitialStateConfig(args, messages):
    """Helper function used for creating InitialStateConfig."""
    initial_state_config = messages.InitialStateConfig()
    has_set = False
    if args.platform_key_file:
        file_content_buffer = CreateFileContentBuffer(messages, args.platform_key_file)
        initial_state_config.pk = file_content_buffer
        has_set = True
    key_exchange_key_file_paths = getattr(args, 'key_exchange_key_file', [])
    if key_exchange_key_file_paths:
        for file_path in key_exchange_key_file_paths:
            file_content_buffer = CreateFileContentBuffer(messages, file_path)
            initial_state_config.keks.append(file_content_buffer)
            has_set = True
    signature_database_file_paths = getattr(args, 'signature_database_file', [])
    if signature_database_file_paths:
        for file_path in signature_database_file_paths:
            file_content_buffer = CreateFileContentBuffer(messages, file_path)
            initial_state_config.dbs.append(file_content_buffer)
            has_set = True
    forbidden_signature_database_file_paths = getattr(args, 'forbidden_database_file', [])
    if forbidden_signature_database_file_paths:
        for file_path in forbidden_signature_database_file_paths:
            file_content_buffer = CreateFileContentBuffer(messages, file_path)
            initial_state_config.dbxs.append(file_content_buffer)
            has_set = True
    return (initial_state_config, has_set)