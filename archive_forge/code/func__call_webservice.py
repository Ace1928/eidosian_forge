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
def _call_webservice(call, *args, **kwargs):
    """Make a call to the webservice, wrapping failures.

    :param call: The call to make.
    :param *args: *args for the call.
    :param **kwargs: **kwargs for the call.
    :return: The result of calling call(*args, *kwargs).
    """
    from lazr.restfulclient import errors as restful_errors
    try:
        return call(*args, **kwargs)
    except restful_errors.HTTPError as e:
        error_lines = []
        for line in e.content.splitlines():
            if line.startswith(b'Traceback (most recent call last):'):
                break
            error_lines.append(line)
        raise WebserviceFailure(b''.join(error_lines))