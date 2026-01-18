from urllib import parse
from zunclient import api_versions
from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
def _process_mounts(self, kwargs):
    mounts = kwargs.get('mounts', None)
    if mounts:
        for mount in mounts:
            if mount.get('type') == 'bind':
                mount['source'] = utils.encode_file_data(mount['source'])