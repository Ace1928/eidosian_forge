import argparse
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from aodhclient import exceptions
from aodhclient.i18n import _
from aodhclient import utils
def _format_alarm(alarm):
    if alarm.get('composite_rule'):
        composite_rule = jsonutils.dumps(alarm['composite_rule'], indent=2)
        alarm['composite_rule'] = composite_rule
        return alarm
    for alarm_type in ALARM_TYPES:
        if alarm.get('%s_rule' % alarm_type):
            alarm.update(alarm.pop('%s_rule' % alarm_type))
    if alarm['time_constraints']:
        alarm['time_constraints'] = jsonutils.dumps(alarm['time_constraints'], sort_keys=True, indent=2)
    if isinstance(alarm.get('query'), list):
        query_rows = []
        for q in alarm['query']:
            op = ALARM_OP_MAP.get(q['op'], q['op'])
            query_rows.append('%s %s %s' % (q['field'], op, q['value']))
        alarm['query'] = ' AND\n'.join(query_rows)
    return alarm