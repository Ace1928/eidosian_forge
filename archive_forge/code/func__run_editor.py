import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
def _run_editor(filename):
    """Try to execute an editor to edit the commit message."""
    for candidate, candidate_source in _get_editor():
        edargs = cmdline.split(candidate)
        try:
            x = call(edargs + [filename])
        except OSError as e:
            if candidate_source is not None:
                trace.warning('Could not start editor "%s" (specified by %s): %s\n' % (candidate, candidate_source, str(e)))
            continue
            raise
        if x == 0:
            return True
        elif x == 127:
            continue
        else:
            break
    raise BzrError('Could not start any editor.\nPlease specify one with:\n - $BRZ_EDITOR\n - editor=/some/path in %s\n - $VISUAL\n - $EDITOR' % bedding.config_path())