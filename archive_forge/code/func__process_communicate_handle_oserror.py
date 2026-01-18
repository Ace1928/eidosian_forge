import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _process_communicate_handle_oserror(process, data, files):
    """Wrapper around process.communicate that checks for OSError."""
    try:
        output, err = process.communicate(data)
    except OSError as e:
        if e.errno != errno.EPIPE:
            raise
        retcode, err = _check_files_accessible(files)
        if process.stderr:
            msg = process.stderr.read()
            if isinstance(msg, bytes):
                msg = msg.decode('utf-8')
            if err:
                err = _('Hit OSError in _process_communicate_handle_oserror(): %(stderr)s\nLikely due to %(file)s: %(error)s') % {'stderr': msg, 'file': err[0], 'error': err[1]}
            else:
                err = _('Hit OSError in _process_communicate_handle_oserror(): %s') % msg
        output = ''
    else:
        retcode = process.poll()
        if err is not None:
            if isinstance(err, bytes):
                err = err.decode('utf-8')
    return (output, err, retcode)