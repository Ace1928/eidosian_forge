from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def ParseKerberosConfigFile(dataproc, kerberos_config_file):
    """Parse a kerberos-config-file into the KerberosConfig message."""
    data = console_io.ReadFromFileOrStdin(kerberos_config_file, binary=False)
    try:
        kerberos_config_data = yaml.load(data)
    except Exception as e:
        raise exceptions.ParseError('Cannot parse YAML:[{0}]'.format(e))
    ssl_config = kerberos_config_data.get('ssl', {})
    keystore_uri = ssl_config.get('keystore_uri')
    truststore_uri = ssl_config.get('truststore_uri')
    keystore_password_uri = ssl_config.get('keystore_password_uri')
    key_password_uri = ssl_config.get('key_password_uri')
    truststore_password_uri = ssl_config.get('truststore_password_uri')
    cross_realm_trust_config = kerberos_config_data.get('cross_realm_trust', {})
    cross_realm_trust_realm = cross_realm_trust_config.get('realm')
    cross_realm_trust_kdc = cross_realm_trust_config.get('kdc')
    cross_realm_trust_admin_server = cross_realm_trust_config.get('admin_server')
    cross_realm_trust_shared_password_uri = cross_realm_trust_config.get('shared_password_uri')
    kerberos_config_msg = dataproc.messages.KerberosConfig(enableKerberos=kerberos_config_data.get('enable_kerberos', True), rootPrincipalPasswordUri=kerberos_config_data.get('root_principal_password_uri'), kmsKeyUri=kerberos_config_data.get('kms_key_uri'), kdcDbKeyUri=kerberos_config_data.get('kdc_db_key_uri'), tgtLifetimeHours=kerberos_config_data.get('tgt_lifetime_hours'), realm=kerberos_config_data.get('realm'), keystoreUri=keystore_uri, keystorePasswordUri=keystore_password_uri, keyPasswordUri=key_password_uri, truststoreUri=truststore_uri, truststorePasswordUri=truststore_password_uri, crossRealmTrustRealm=cross_realm_trust_realm, crossRealmTrustKdc=cross_realm_trust_kdc, crossRealmTrustAdminServer=cross_realm_trust_admin_server, crossRealmTrustSharedPasswordUri=cross_realm_trust_shared_password_uri)
    return kerberos_config_msg