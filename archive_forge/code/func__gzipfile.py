import gzip
import os
import shutil
import zipfile
from oslo_log import log as logging
from oslo_utils import encodeutils
from taskflow.patterns import linear_flow as lf
from taskflow import task
def _gzipfile(src_path, dest_path, image_id):
    try:
        with gzip.open(src_path, 'r') as gzfd:
            with open(dest_path, 'wb') as fd:
                shutil.copyfileobj(gzfd, fd)
    except gzip.BadGzipFile as e:
        LOG.debug('ZIP: Error decompressing image %(iid)s: Bad GZip file: %(msg)s', {'iid': image_id, 'msg': encodeutils.exception_to_unicode(e)})
        raise
    except Exception as e:
        LOG.debug('GZIP: Error decompressing image %(iid)s: %(msg)s', {'iid': image_id, 'msg': encodeutils.exception_to_unicode(e)})
        raise