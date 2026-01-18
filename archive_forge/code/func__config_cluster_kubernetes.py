import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography import x509
from cryptography.x509.oid import NameOID
from oslo_serialization import base64
from oslo_serialization import jsonutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
def _config_cluster_kubernetes(cluster, cluster_template, cfg_dir, force=False, certs=None, use_keystone=False, direct_output=False):
    """Return and write configuration for the given kubernetes cluster."""
    cfg_file = '%s/config' % cfg_dir
    if cluster_template.tls_disabled or certs is None:
        cfg = "apiVersion: v1\nclusters:\n- cluster:\n    server: %(api_address)s\n  name: %(name)s\ncontexts:\n- context:\n    cluster: %(name)s\n    user: %(name)s\n  name: %(name)s\ncurrent-context: %(name)s\nkind: Config\npreferences: {}\nusers:\n- name: %(name)s'\n" % {'name': cluster.name, 'api_address': cluster.api_address}
    elif not use_keystone:
        cfg = 'apiVersion: v1\nclusters:\n- cluster:\n    certificate-authority-data: %(ca)s\n    server: %(api_address)s\n  name: %(name)s\ncontexts:\n- context:\n    cluster: %(name)s\n    user: admin\n  name: default\ncurrent-context: default\nkind: Config\npreferences: {}\nusers:\n- name: admin\n  user:\n    client-certificate-data: %(cert)s\n    client-key-data: %(key)s\n' % {'name': cluster.name, 'api_address': cluster.api_address, 'key': base64.encode_as_text(certs['key']), 'cert': base64.encode_as_text(certs['cert']), 'ca': base64.encode_as_text(certs['ca'])}
    else:
        cfg = 'apiVersion: v1\nclusters:\n- cluster:\n    certificate-authority-data: %(ca)s\n    server: %(api_address)s\n  name: %(name)s\ncontexts:\n- context:\n    cluster: %(name)s\n    user: openstackuser\n  name: openstackuser@kubernetes\ncurrent-context: openstackuser@kubernetes\nkind: Config\npreferences: {}\nusers:\n- name: openstackuser\n  user:\n    exec:\n      command: /bin/bash\n      apiVersion: client.authentication.k8s.io/v1beta1\n      args:\n      - -c\n      - >\n        if [ -z ${OS_TOKEN} ]; then\n            echo \'Error: Missing OpenStack credential from environment variable $OS_TOKEN\' > /dev/stderr\n            exit 1\n        else\n            echo \'{ "apiVersion": "client.authentication.k8s.io/v1beta1", "kind": "ExecCredential", "status": { "token": "\'"${OS_TOKEN}"\'"}}\'\n        fi\n' % {'name': cluster.name, 'api_address': cluster.api_address, 'ca': base64.encode_as_text(certs['ca'])}
    if direct_output:
        return cfg
    if os.path.exists(cfg_file) and (not force):
        raise exc.CommandError('File %s exists, aborting.' % cfg_file)
    else:
        f = open(cfg_file, 'w')
        f.write(cfg)
        f.close()
    if 'csh' in os.environ['SHELL']:
        return 'setenv KUBECONFIG %s\n' % cfg_file
    else:
        return 'export KUBECONFIG=%s\n' % cfg_file