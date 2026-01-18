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
def _ParseTokenArgs(self, token_type, cluster, aws_sts_region, id_token, access_token, access_token_expiry, refresh_token, client_id, client_secret, idp_certificate_authority_data, idp_issuer_url, kubeconfig_path, user, **kwargs):
    del kwargs
    exec_args = ['token']
    if token_type:
        exec_args.extend(['--type', token_type])
    if cluster:
        exec_args.extend(['--cluster', cluster])
    if aws_sts_region:
        exec_args.extend(['--aws-sts-region', aws_sts_region])
    if id_token:
        exec_args.extend(['--id-token', id_token])
    if access_token:
        exec_args.extend(['--access-token', access_token])
    if access_token_expiry:
        exec_args.extend(['--access-token-expiry', access_token_expiry])
    if refresh_token:
        exec_args.extend(['--refresh-token', refresh_token])
    if client_id:
        exec_args.extend(['--client-id', client_id])
    if client_secret:
        exec_args.extend(['--client-secret', client_secret])
    if idp_certificate_authority_data:
        exec_args.extend(['--idp-certificate-authority-data', idp_certificate_authority_data])
    if idp_issuer_url:
        exec_args.extend(['--idp-issuer-url', idp_issuer_url])
    if kubeconfig_path:
        exec_args.extend(['--kubeconfig-path', kubeconfig_path])
    if user:
        exec_args.extend(['--user', user])
    return exec_args