import argparse
import copy
import getpass
import hashlib
import json
import logging
import os
import sys
import traceback
from oslo_utils import encodeutils
from oslo_utils import importutils
import urllib.parse
import glanceclient
from glanceclient._i18n import _
from glanceclient.common import utils
from glanceclient import exc
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import loading
def _cache_schemas(self, options, client, home_dir='~/.glanceclient'):
    homedir = os.path.expanduser(home_dir)
    path_prefix = homedir
    if options.os_auth_url:
        hash_host = hashlib.sha1(options.os_auth_url.encode('utf-8'))
        path_prefix = os.path.join(path_prefix, hash_host.hexdigest())
    if not os.path.exists(path_prefix):
        try:
            os.makedirs(path_prefix)
        except OSError as e:
            msg = '%s' % e
            print(encodeutils.safe_decode(msg), file=sys.stderr)
    resources = ['image', 'metadefs/namespace', 'metadefs/resource_type']
    schema_file_paths = [os.path.join(path_prefix, x + '_schema.json') for x in ['image', 'namespace', 'resource_type']]
    failed_download_schema = 0
    for resource, schema_file_path in zip(resources, schema_file_paths):
        if not os.path.exists(schema_file_path) or options.get_schema:
            try:
                schema = client.schemas.get(resource)
                with open(schema_file_path, 'w') as f:
                    f.write(json.dumps(schema.raw()))
            except exc.Unauthorized:
                raise exc.CommandError('Invalid OpenStack Identity credentials.')
            except Exception:
                failed_download_schema += 1
                pass
    return failed_download_schema >= len(resources)