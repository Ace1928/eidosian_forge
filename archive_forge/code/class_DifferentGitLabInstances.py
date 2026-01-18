import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
class DifferentGitLabInstances(errors.BzrError):
    _fmt = "Can't create merge proposals across GitLab instances: %(source_host)s and %(target_host)s"

    def __init__(self, source_host, target_host):
        self.source_host = source_host
        self.target_host = target_host