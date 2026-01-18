import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def _parse_alerts_blob_key(blob_key):
    """Parse the BLOB key for Alerts.

    Args:
      blob_key: The BLOB key to parse. By contract, it should have the format:
       - `${ALERTS_BLOB_TAG_PREFIX}_${begin}_${end}.${run_id}` when there is no
         alert type filter.
       - `${ALERTS_BLOB_TAG_PREFIX}_${begin}_${end}_${alert_filter}.${run_id}`
         when there is an alert type filter.

    Returns:
      - run ID
      - begin index
      - end index
      - alert_type: alert type string used to filter retrieved alert data.
          `None` if no filtering is used.
    """
    key_body, run = blob_key.split('.', 1)
    key_body = key_body[len(ALERTS_BLOB_TAG_PREFIX):]
    key_items = key_body.split('_', 3)
    begin = int(key_items[1])
    end = int(key_items[2])
    alert_type = None
    if len(key_items) > 3:
        alert_type = key_items[3]
    return (run, begin, end, alert_type)