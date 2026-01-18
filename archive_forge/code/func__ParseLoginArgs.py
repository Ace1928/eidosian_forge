from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import copy
import json
import os
from googlecloudsdk.command_lib.anthos import flags
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
from six.moves import urllib
def _ParseLoginArgs(self, cluster, kube_config=None, login_config=None, login_config_cert=None, user=None, ldap_user=None, ldap_pass=None, dry_run=None, preferred_auth=None, server_url=None, **kwargs):
    del kwargs
    exec_args = ['login']
    if cluster:
        exec_args.extend(['--cluster', cluster])
    if kube_config:
        exec_args.extend(['--kubeconfig', kube_config])
    if login_config:
        exec_args.extend(['--login-config', login_config])
    if login_config_cert:
        exec_args.extend(['--login-config-cert', login_config_cert])
    if user:
        exec_args.extend(['--user', user])
    if dry_run:
        exec_args.extend(['--dry-run'])
    if ldap_pass and ldap_user:
        exec_args.extend(['--ldap-username', ldap_user, '--ldap-password', ldap_pass])
    if preferred_auth:
        exec_args.extend(['--preferred-auth', preferred_auth])
    if server_url:
        exec_args.extend(['--server', server_url])
    return exec_args