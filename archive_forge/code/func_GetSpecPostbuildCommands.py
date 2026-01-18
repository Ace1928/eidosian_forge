import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetSpecPostbuildCommands(spec, quiet=False):
    """Returns the list of postbuilds explicitly defined on |spec|, in a form
  executable by a shell."""
    postbuilds = []
    for postbuild in spec.get('postbuilds', []):
        if not quiet:
            postbuilds.append('echo POSTBUILD\\(%s\\) %s' % (spec['target_name'], postbuild['postbuild_name']))
        postbuilds.append(gyp.common.EncodePOSIXShellList(postbuild['action']))
    return postbuilds