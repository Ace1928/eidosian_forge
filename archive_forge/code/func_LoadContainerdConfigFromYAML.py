from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def LoadContainerdConfigFromYAML(containerd_config, content, messages):
    """Load containerd configuration from YAML/JSON file.

  Args:
    containerd_config: The containerd config object to be populated (either from
      a node or from node config defaults).
    content: The YAML/JSON string that contains private CR config.
    messages: The message module.

  Raises:
    Error: when there's any errors on parsing the YAML/JSON system config.
  """
    try:
        opts = yaml.load(content)
    except yaml.YAMLParseError as e:
        raise NodeConfigError('config is not valid YAML/JSON: {0}'.format(e))
    _CheckNodeConfigFields('<root>', opts, {NC_CC_PRIVATE_CR_CONFIG: dict})
    private_registry_opts = opts.get(NC_CC_PRIVATE_CR_CONFIG)
    if private_registry_opts:
        config_fields = {NC_CC_PRIVATE_CR_CONFIG_ENABLED: bool, NC_CC_CA_CONFIG: list}
        _CheckNodeConfigFields(NC_CC_PRIVATE_CR_CONFIG, private_registry_opts, config_fields)
        containerd_config.privateRegistryAccessConfig = messages.PrivateRegistryAccessConfig()
        containerd_config.privateRegistryAccessConfig.enabled = private_registry_opts.get(NC_CC_PRIVATE_CR_CONFIG_ENABLED)
        ca_domain_opts = private_registry_opts.get(NC_CC_CA_CONFIG)
        if ca_domain_opts:
            config_fields = {NC_CC_GCP_SECRET_CONFIG: dict, NC_CC_PRIVATE_CR_FQDNS_CONFIG: list}
            containerd_config.privateRegistryAccessConfig.certificateAuthorityDomainConfig = []
            for i, opts in enumerate(ca_domain_opts):
                _CheckNodeConfigFields('{0}[{1}]'.format(NC_CC_CA_CONFIG, i), opts, config_fields)
                gcp_secret_opts = opts.get(NC_CC_GCP_SECRET_CONFIG)
                if not gcp_secret_opts:
                    raise NodeConfigError('privateRegistryAccessConfig.certificateAuthorityDomainConfig must specify a secret config, none was provided')
                _CheckNodeConfigFields(NC_CC_GCP_SECRET_CONFIG, gcp_secret_opts, {NC_CC_GCP_SECRET_CONFIG_SECRET_URI: str})
                ca_config = messages.CertificateAuthorityDomainConfig()
                ca_config.gcpSecretManagerCertificateConfig = messages.GCPSecretManagerCertificateConfig()
                ca_config.gcpSecretManagerCertificateConfig.secretUri = gcp_secret_opts.get(NC_CC_GCP_SECRET_CONFIG_SECRET_URI)
                ca_config.fqdns = opts.get(NC_CC_PRIVATE_CR_FQDNS_CONFIG)
                containerd_config.privateRegistryAccessConfig.certificateAuthorityDomainConfig.append(ca_config)