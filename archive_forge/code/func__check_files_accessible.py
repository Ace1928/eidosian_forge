import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _check_files_accessible(files):
    err = None
    retcode = -1
    try:
        for try_file in files:
            with open(try_file, 'r'):
                pass
    except IOError as e:
        err = (try_file, e.strerror)
        retcode = OpensslCmsExitStatus.INPUT_FILE_READ_ERROR
    return (retcode, err)