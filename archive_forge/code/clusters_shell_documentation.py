import os
from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
Configure native client to access cluster.

    You can source the output of this command to get the native client of the
    corresponding COE configured to access the cluster.

    Example: eval $(magnum cluster-config <cluster-name>).
    