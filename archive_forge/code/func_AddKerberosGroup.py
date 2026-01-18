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
def AddKerberosGroup(parser):
    """Adds the argument group to handle Kerberos configurations."""
    kerberos_group = parser.add_argument_group(mutex=True, help='Specifying these flags will enable Kerberos for the cluster.')
    kerberos_flag_group = kerberos_group.add_argument_group()
    kerberos_flag_group.add_argument('--enable-kerberos', action='store_true', help='        Enable Kerberos on the cluster.\n        ')
    kerberos_flag_group.add_argument('--kerberos-root-principal-password-uri', help="        Google Cloud Storage URI of a KMS encrypted file containing the root\n        principal password. Must be a Cloud Storage URL beginning with 'gs://'.\n        ")
    kerberos_kms_flag_overrides = {'kms-key': '--kerberos-kms-key', 'kms-keyring': '--kerberos-kms-key-keyring', 'kms-location': '--kerberos-kms-key-location', 'kms-project': '--kerberos-kms-key-project'}
    kms_resource_args.AddKmsKeyResourceArg(kerberos_flag_group, 'password', flag_overrides=kerberos_kms_flag_overrides, name='--kerberos-kms-key')
    kerberos_group.add_argument('--kerberos-config-file', help='Path to a YAML (or JSON) file containing the configuration for Kerberos on the\ncluster. If you pass `-` as the value of the flag the file content will be read\nfrom stdin.\n\nThe YAML file is formatted as follows:\n\n```\n  # Optional. Flag to indicate whether to Kerberize the cluster.\n  # The default value is true.\n  enable_kerberos: true\n\n  # Optional. The Google Cloud Storage URI of a KMS encrypted file\n  # containing the root principal password.\n  root_principal_password_uri: gs://bucket/password.encrypted\n\n  # Optional. The URI of the Cloud KMS key used to encrypt\n  # sensitive files.\n  kms_key_uri:\n    projects/myproject/locations/global/keyRings/mykeyring/cryptoKeys/my-key\n\n  # Configuration of SSL encryption. If specified, all sub-fields\n  # are required. Otherwise, Dataproc will provide a self-signed\n  # certificate and generate the passwords.\n  ssl:\n    # Optional. The Google Cloud Storage URI of the keystore file.\n    keystore_uri: gs://bucket/keystore.jks\n\n    # Optional. The Google Cloud Storage URI of a KMS encrypted\n    # file containing the password to the keystore.\n    keystore_password_uri: gs://bucket/keystore_password.encrypted\n\n    # Optional. The Google Cloud Storage URI of a KMS encrypted\n    # file containing the password to the user provided key.\n    key_password_uri: gs://bucket/key_password.encrypted\n\n    # Optional. The Google Cloud Storage URI of the truststore\n    # file.\n    truststore_uri: gs://bucket/truststore.jks\n\n    # Optional. The Google Cloud Storage URI of a KMS encrypted\n    # file containing the password to the user provided\n    # truststore.\n    truststore_password_uri:\n      gs://bucket/truststore_password.encrypted\n\n  # Configuration of cross realm trust.\n  cross_realm_trust:\n    # Optional. The remote realm the Dataproc on-cluster KDC will\n    # trust, should the user enable cross realm trust.\n    realm: REMOTE.REALM\n\n    # Optional. The KDC (IP or hostname) for the remote trusted\n    # realm in a cross realm trust relationship.\n    kdc: kdc.remote.realm\n\n    # Optional. The admin server (IP or hostname) for the remote\n    # trusted realm in a cross realm trust relationship.\n    admin_server: admin-server.remote.realm\n\n    # Optional. The Google Cloud Storage URI of a KMS encrypted\n    # file containing the shared password between the on-cluster\n    # Kerberos realm and the remote trusted realm, in a cross\n    # realm trust relationship.\n    shared_password_uri:\n      gs://bucket/cross-realm.password.encrypted\n\n  # Optional. The Google Cloud Storage URI of a KMS encrypted file\n  # containing the master key of the KDC database.\n  kdc_db_key_uri: gs://bucket/kdc_db_key.encrypted\n\n  # Optional. The lifetime of the ticket granting ticket, in\n  # hours. If not specified, or user specifies 0, then default\n  # value 10 will be used.\n  tgt_lifetime_hours: 1\n\n  # Optional. The name of the Kerberos realm. If not specified,\n  # the uppercased domain name of the cluster will be used.\n  realm: REALM.NAME\n```\n        ')