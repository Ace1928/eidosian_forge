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
class NotGitLabUrl(errors.BzrError):
    _fmt = 'Not a GitLab URL: %(url)s'

    def __init__(self, url):
        errors.BzrError.__init__(self)
        self.url = url