import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def _get_cmd_upload(self):
    cmd = cmds.cmd_upload()
    cmd.outf = sys.stdout
    return cmd