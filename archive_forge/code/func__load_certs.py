import os
import json
from shutil import copyfile, rmtree
from docker.tls import TLSConfig
from docker.errors import ContextException
from docker.context.config import get_meta_dir
from docker.context.config import get_meta_file
from docker.context.config import get_tls_dir
from docker.context.config import get_context_host
def _load_certs(self):
    certs = {}
    tls_dir = get_tls_dir(self.name)
    for endpoint in self.endpoints.keys():
        if not os.path.isdir(os.path.join(tls_dir, endpoint)):
            continue
        ca_cert = None
        cert = None
        key = None
        for filename in os.listdir(os.path.join(tls_dir, endpoint)):
            if filename.startswith('ca'):
                ca_cert = os.path.join(tls_dir, endpoint, filename)
            elif filename.startswith('cert'):
                cert = os.path.join(tls_dir, endpoint, filename)
            elif filename.startswith('key'):
                key = os.path.join(tls_dir, endpoint, filename)
        if all([ca_cert, cert, key]):
            verify = None
            if endpoint == 'docker' and (not self.endpoints['docker'].get('SkipTLSVerify', False)):
                verify = True
            certs[endpoint] = TLSConfig(client_cert=(cert, key), ca_cert=ca_cert, verify=verify)
    self.tls_cfg = certs
    self.tls_path = tls_dir