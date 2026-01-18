from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class LbaasL7RuleMixin(object):

    def set_extra_attrs(self, parsed_args):
        self.parent_id = _get_policy_id(self.get_client(), parsed_args.l7policy)

    def add_known_arguments(self, parser):
        parser.add_argument('l7policy', metavar='L7POLICY', help=_('ID or name of L7 policy this rule belongs to.'))