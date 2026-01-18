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
def _find_alarm_by_name(client, name):
    query = jsonutils.dumps({'=': {'name': name}})
    alarms = client.alarm.query(query)
    if len(alarms) > 1:
        msg = _("Multiple alarms matches found for '%s', use an ID to be more specific.") % name
        raise exceptions.NoUniqueMatch(msg)
    elif not alarms:
        msg = _('Alarm %s not found') % name
        raise exceptions.NotFound(msg)
    else:
        return alarms[0]