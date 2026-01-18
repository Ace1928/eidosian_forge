import argparse
import itertools
import json
import logging
import sys
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class ProvisionStateWithWait(ProvisionStateBaremetalNode):
    """Provision state class adding --wait flag."""
    log = logging.getLogger(__name__ + '.ProvisionStateWithWait')

    def get_parser(self, prog_name):
        parser = super(ProvisionStateWithWait, self).get_parser(prog_name)
        desired_state = v1_utils.PROVISION_ACTIONS.get(self.PROVISION_STATE)['expected_state']
        parser.add_argument('--wait', type=int, dest='wait_timeout', default=None, metavar='<time-out>', const=0, nargs='?', help=_('Wait for a node to reach the desired state, %(state)s. Optionally takes a timeout value (in seconds). The default value is 0, meaning it will wait indefinitely.') % {'state': desired_state})
        return parser

    def take_action(self, parsed_args):
        super(ProvisionStateWithWait, self).take_action(parsed_args)
        self.log.debug('take_action(%s)', parsed_args)
        if parsed_args.wait_timeout is None:
            return
        baremetal_client = self.app.client_manager.baremetal
        wait_args = v1_utils.PROVISION_ACTIONS.get(parsed_args.provision_state)
        if wait_args is None:
            raise exc.CommandError(_("'--wait is not supported for provision state '%s'") % parsed_args.provision_state)
        print(_('Waiting for provision state %(state)s on node(s) %(node)s') % {'state': wait_args['expected_state'], 'node': ', '.join(parsed_args.nodes)})
        baremetal_client.node.wait_for_provision_state(parsed_args.nodes, timeout=parsed_args.wait_timeout, **wait_args)