from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_template_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import partner_metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import resource_manager_tags_utils
from googlecloudsdk.command_lib.compute.instance_templates import flags as instance_templates_flags
from googlecloudsdk.command_lib.compute.instance_templates import mesh_util
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import flags as sole_tenancy_flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
import six
def AddServiceProxyArgsToMetadata(args):
    """Inserts the Service Proxy arguments provided by the user to the instance metadata.

  Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the .Args() method.
  """
    if getattr(args, 'service_proxy', False):
        service_proxy_config = collections.OrderedDict()
        proxy_spec = collections.OrderedDict()
        service_proxy_config['_disclaimer'] = service_proxy_aux_data.DISCLAIMER
        service_proxy_config['api-version'] = '0.2'
        if 'serving-ports' in args.service_proxy:
            serving_ports = list(map(int, args.service_proxy['serving-ports'].split(';')))
            unique_serving_ports = set(serving_ports)
            serving_ports = list(unique_serving_ports)
            service_proxy_config['service'] = {'serving-ports': serving_ports}
        if 'proxy-port' in args.service_proxy:
            proxy_spec['proxy-port'] = args.service_proxy['proxy-port']
        if 'tracing' in args.service_proxy:
            proxy_spec['tracing'] = args.service_proxy['tracing']
        if 'access-log' in args.service_proxy:
            proxy_spec['access-log'] = args.service_proxy['access-log']
        proxy_spec['network'] = args.service_proxy.get('network', '')
        if 'scope' in args.service_proxy:
            proxy_spec['scope'] = args.service_proxy['scope']
        if 'mesh' in args.service_proxy:
            proxy_spec['mesh'] = args.service_proxy['mesh']
        if 'project-number' in args.service_proxy:
            proxy_spec['project-number'] = args.service_proxy['project-number']
        if 'source' in args.service_proxy:
            proxy_spec['primary-source'] = args.service_proxy['source']
            proxy_spec['secondary-source'] = args.service_proxy['source']
        traffic_interception = collections.OrderedDict()
        if 'intercept-all-outbound-traffic' in args.service_proxy:
            traffic_interception['intercept-all-outbound'] = True
            if 'exclude-outbound-ip-ranges' in args.service_proxy:
                traffic_interception['exclude-outbound-ip-ranges'] = args.service_proxy['exclude-outbound-ip-ranges'].split(';')
            if 'exclude-outbound-port-ranges' in args.service_proxy:
                traffic_interception['exclude-outbound-port-ranges'] = args.service_proxy['exclude-outbound-port-ranges'].split(';')
        if 'intercept-dns' in args.service_proxy:
            traffic_interception['intercept-dns'] = True
        if traffic_interception:
            service_proxy_config['traffic-interception'] = traffic_interception
        if getattr(args, 'service_proxy_xds_version', False):
            proxy_spec['xds-version'] = args.service_proxy_xds_version
        if getattr(args, 'service_proxy_labels', False):
            service_proxy_config['labels'] = args.service_proxy_labels
        args.metadata['enable-osconfig'] = 'true'
        gce_software_declaration = collections.OrderedDict()
        service_proxy_agent_recipe = collections.OrderedDict()
        service_proxy_agent_recipe['name'] = 'install-gce-service-proxy-agent'
        service_proxy_agent_recipe['desired_state'] = 'INSTALLED'
        if getattr(args, 'service_proxy_agent_location', False):
            service_proxy_agent_recipe['installSteps'] = [{'scriptRun': {'script': service_proxy_aux_data.startup_script_with_location_template % args.service_proxy_agent_location}}]
        else:
            service_proxy_agent_recipe['installSteps'] = [{'scriptRun': {'script': service_proxy_aux_data.startup_script}}]
        gce_software_declaration['softwareRecipes'] = [service_proxy_agent_recipe]
        args.metadata['gce-software-declaration'] = json.dumps(gce_software_declaration)
        args.metadata['enable-guest-attributes'] = 'TRUE'
        if proxy_spec:
            service_proxy_config['proxy-spec'] = proxy_spec
        args.metadata['gce-service-proxy'] = json.dumps(service_proxy_config)