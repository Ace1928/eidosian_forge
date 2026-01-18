import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def _parse_graph_op_info_blob_key(blob_key):
    """Parse the BLOB key for graph op info.

    Args:
      blob_key: The BLOB key to parse. By contract, it should have the format:
       `${GRAPH_OP_INFO_BLOB_TAG_PREFIX}_${graph_id}_${op_name}.${run_name}`,
      wherein
        - `graph_id` is a UUID
        - op_name conforms to the TensorFlow spec:
          `^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$`
        - `run_name` is assumed to contain no dots (`'.'`s).

    Returns:
      - run name
      - graph_id
      - op name
    """
    last_dot_index = blob_key.rindex('.')
    run = blob_key[last_dot_index + 1:]
    key_body = blob_key[:last_dot_index]
    key_body = key_body[len(GRAPH_OP_INFO_BLOB_TAG_PREFIX):]
    _, graph_id, op_name = key_body.split('_', 2)
    return (run, graph_id, op_name)