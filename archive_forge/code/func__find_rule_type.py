import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def _find_rule_type(qos, rule_id):
    for rule in (r for r in qos.rules if r['id'] == rule_id):
        return rule['type'].replace('_', '-')
    return None