import re
import shutil
import tempfile
from typing import Any, List, Optional
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ...forge import (AutoMergeUnsupported, Forge, LabelsUnsupported,
from ...git.urls import git_url_to_bzr_url
from ...lazy_import import lazy_import
from ...trace import mutter
from breezy.plugins.launchpad import (
from ...transport import get_transport
def _split_url(self, url):
    url, params = urlutils.split_segment_parameters(url)
    scheme, user, password, host, port, path = urlutils.parse_url(url)
    path = path.strip('/')
    if host.startswith('bazaar.'):
        vcs = 'bzr'
    elif host.startswith('git.'):
        vcs = 'git'
    else:
        raise ValueError('unknown host %s' % host)
    return (vcs, user, password, path, params)