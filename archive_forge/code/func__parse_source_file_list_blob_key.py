import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def _parse_source_file_list_blob_key(blob_key):
    """Parse the BLOB key for source file list.

    Args:
      blob_key: The BLOB key to parse. By contract, it should have the format:
       `${SOURCE_FILE_LIST_BLOB_TAG}.${run_id}`

    Returns:
      - run ID
    """
    return blob_key[blob_key.index('.') + 1:]