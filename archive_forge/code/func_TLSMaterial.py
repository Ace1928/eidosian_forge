import os
import json
from shutil import copyfile, rmtree
from docker.tls import TLSConfig
from docker.errors import ContextException
from docker.context.config import get_meta_dir
from docker.context.config import get_meta_file
from docker.context.config import get_tls_dir
from docker.context.config import get_context_host
@property
def TLSMaterial(self):
    certs = {}
    for endpoint, tls in self.tls_cfg.items():
        cert, key = tls.cert
        certs[endpoint] = list(map(os.path.basename, [tls.ca_cert, cert, key]))
    return {'TLSMaterial': certs}