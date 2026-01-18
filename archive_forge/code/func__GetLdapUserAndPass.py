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
def _GetLdapUserAndPass(cluster_config, auth_name, cluster):
    """Prompt User for Ldap Username and Password."""
    ldap_user = None
    ldap_pass = None
    if not cluster_config.IsLdap():
        return (None, None)
    user_message = 'Please enter the ldap user for [{}] on cluster [{}]: '.format(auth_name, cluster)
    pass_message = 'Please enter the ldap password for [{}] on cluster [{}]: '.format(auth_name, cluster)
    ldap_user = console_io.PromptWithValidator(validator=lambda x: len(x) > 1, error_message='Error: Invalid username, please try again.', prompt_string=user_message)
    ldap_pass = console_io.PromptPassword(pass_message, validation_callable=lambda x: len(x) > 1)
    return _Base64EncodeLdap(ldap_user, ldap_pass)