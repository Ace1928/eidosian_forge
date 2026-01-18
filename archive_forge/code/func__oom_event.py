from concurrent import futures
import datetime
import json
import logging
import os
import time
import urllib
from absl import flags
def _oom_event(self, symptoms):
    """Check if a runtime OOM event is reported."""
    if not symptoms:
        return False
    for symptom in reversed(symptoms):
        if symptom['symptomType'] != 'OUT_OF_MEMORY':
            continue
        oom_datetime_str = symptom['createTime'].split('.')[0]
        oom_datetime = datetime.datetime.strptime(oom_datetime_str, '%Y-%m-%dT%H:%M:%S')
        time_diff = _utcnow() - oom_datetime
        if time_diff < datetime.timedelta(seconds=_OOM_EVENT_COOL_TIME_SEC):
            logging.warning(self._symptom_msg('a recent runtime OOM has occurred ~{} seconds ago. The model script will terminate automatically. To prevent future OOM events, please consider reducing the model size. To disable this behavior, set flag --runtime_oom_exit=false when starting the script.'.format(time_diff.seconds)))
            return True
    return False