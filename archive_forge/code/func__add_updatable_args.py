from neutronclient._i18n import _
from neutronclient.common import extension
def _add_updatable_args(parser):
    parser.add_argument('name', help=_('Name of this fox socket.'))