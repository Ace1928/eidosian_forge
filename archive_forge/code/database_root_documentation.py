from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from troveclient.i18n import _
Returns an instance or cluster, found by ID or name,
    along with the type of resource, instance or cluster.
    Raises CommandError if none is found.
    