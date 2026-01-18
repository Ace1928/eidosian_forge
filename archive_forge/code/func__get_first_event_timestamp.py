import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def _get_first_event_timestamp(self, run_name):
    try:
        return self._multiplexer.FirstEventTimestamp(run_name)
    except ValueError as e:
        return None