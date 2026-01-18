import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
def _format_sg_rules(secgroup):
    try:
        return '\n'.join(sorted([_format_sg_rule(rule) for rule in secgroup['security_group_rules']]))
    except Exception:
        return ''