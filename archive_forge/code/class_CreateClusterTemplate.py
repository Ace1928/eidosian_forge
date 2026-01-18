from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
class CreateClusterTemplate(command.ShowOne):
    """Create a Cluster Template."""
    _description = _('Create a Cluster Template.')

    def get_parser(self, prog_name):
        parser = super(CreateClusterTemplate, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the cluster template to create.'))
        parser.add_argument('--coe', required=True, metavar='<coe>', help=_('Specify the Container Orchestration Engine to use.'))
        parser.add_argument('--image', required=True, metavar='<image>', help=_('The name or UUID of the base image to customize for the Cluster.'))
        parser.add_argument('--external-network', dest='external_network', required=True, metavar='<external-network>', help=_('The external Neutron network name or UUID to connect to this Cluster Template.'))
        parser.add_argument('--keypair', metavar='<keypair>', help=_('The name or UUID of the SSH keypair to load into the Cluster nodes.'))
        parser.add_argument('--fixed-network', dest='fixed_network', metavar='<fixed-network>', help=_('The private Neutron network name to connect to this Cluster model.'))
        parser.add_argument('--fixed-subnet', dest='fixed_subnet', metavar='<fixed-subnet>', help=_('The private Neutron subnet name to connect to Cluster.'))
        parser.add_argument('--network-driver', dest='network_driver', metavar='<network-driver>', help=_('The network driver name for instantiating container networks.'))
        parser.add_argument('--volume-driver', dest='volume_driver', metavar='<volume-driver>', help=_('The volume driver name for instantiating container volume.'))
        parser.add_argument('--dns-nameserver', dest='dns_nameserver', metavar='<dns-nameserver>', default='8.8.8.8', help=_('The DNS nameserver to use for this cluster template.'))
        parser.add_argument('--flavor', metavar='<flavor>', default='m1.medium', help=_('The nova flavor name or UUID to use when launching the Cluster.'))
        parser.add_argument('--master-flavor', dest='master_flavor', metavar='<master-flavor>', help=_('The nova flavor name or UUID to use when launching the master node of the Cluster.'))
        parser.add_argument('--docker-volume-size', dest='docker_volume_size', metavar='<docker-volume-size>', type=int, help=_('Specify the number of size in GB for the docker volume to use.'))
        parser.add_argument('--docker-storage-driver', dest='docker_storage_driver', metavar='<docker-storage-driver>', default='overlay2', help=_('Select a docker storage driver. Supported: devicemapper, overlay, overlay2. Default: overlay2'))
        parser.add_argument('--http-proxy', dest='http_proxy', metavar='<http-proxy>', help=_('The http_proxy address to use for nodes in Cluster.'))
        parser.add_argument('--https-proxy', dest='https_proxy', metavar='<https-proxy>', help=_('The https_proxy address to use for nodes in Cluster.'))
        parser.add_argument('--no-proxy', dest='no_proxy', metavar='<no-proxy>', help=_('The no_proxy address to use for nodes in Cluster.'))
        parser.add_argument('--labels', metavar='<KEY1=VALUE1,KEY2=VALUE2;KEY3=VALUE3...>', action='append', default=[], help=_('Arbitrary labels in the form of key=value pairs to associate with a cluster template. May be used multiple times.'))
        parser.add_argument('--tls-disabled', dest='tls_disabled', action='store_true', default=False, help=_('Disable TLS in the Cluster.'))
        parser.add_argument('--public', action='store_true', default=False, help=_('Make cluster template public.'))
        parser.add_argument('--registry-enabled', dest='registry_enabled', action='store_true', default=False, help=_('Enable docker registry in the Cluster'))
        parser.add_argument('--server-type', dest='server_type', metavar='<server-type>', default='vm', help=_('Specify the server type to be used for example vm. For this release default server type will be vm.'))
        parser.add_argument('--master-lb-enabled', dest='master_lb_enabled', action='store_true', default=False, help=_('Indicates whether created Clusters should have a load balancer for master nodes or not.'))
        parser.add_argument('--floating-ip-enabled', dest='floating_ip_enabled', default=[], action='append_const', const=True, help=_('Indicates whether created Clusters should have a floating ip.'))
        parser.add_argument('--floating-ip-disabled', dest='floating_ip_enabled', action='append_const', const=False, help=_('Disables floating ip creation on the new Cluster'))
        parser.add_argument('--hidden', dest='hidden', action='store_true', default=False, help=_('Indicates the cluster template should be hidden.'))
        parser.add_argument('--visible', dest='hidden', action='store_false', help=_('Indicates the cluster template should be visible.'))
        parser.add_argument('--tags', action='append', default=[], metavar='<--tags tag1 --tags tag2,tag3>', help=_('Tags to be added to the cluster template.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        args = {'name': parsed_args.name, 'image_id': parsed_args.image, 'keypair_id': parsed_args.keypair, 'external_network_id': parsed_args.external_network, 'coe': parsed_args.coe, 'fixed_network': parsed_args.fixed_network, 'fixed_subnet': parsed_args.fixed_subnet, 'network_driver': parsed_args.network_driver, 'volume_driver': parsed_args.volume_driver, 'dns_nameserver': parsed_args.dns_nameserver, 'flavor_id': parsed_args.flavor, 'master_flavor_id': parsed_args.master_flavor, 'docker_volume_size': parsed_args.docker_volume_size, 'docker_storage_driver': parsed_args.docker_storage_driver, 'http_proxy': parsed_args.http_proxy, 'https_proxy': parsed_args.https_proxy, 'no_proxy': parsed_args.no_proxy, 'labels': magnum_utils.handle_labels(parsed_args.labels), 'tls_disabled': parsed_args.tls_disabled, 'public': parsed_args.public, 'registry_enabled': parsed_args.registry_enabled, 'server_type': parsed_args.server_type, 'master_lb_enabled': parsed_args.master_lb_enabled}
        if parsed_args.hidden:
            args['hidden'] = parsed_args.hidden
        if parsed_args.tags:
            args['tags'] = ','.join(set(','.join(parsed_args.tags).split(',')))
        if len(parsed_args.floating_ip_enabled) > 1:
            raise InvalidAttribute('--floating-ip-enabled and --floating-ip-disabled are mutually exclusive and should be specified only once.')
        elif len(parsed_args.floating_ip_enabled) == 1:
            args['floating_ip_enabled'] = parsed_args.floating_ip_enabled[0]
        deprecated = ['devicemapper', 'overlay']
        if args['docker_storage_driver'] in deprecated:
            print('WARNING: Docker storage drivers %s are deprecated and will be removed in a future release. Use overlay2 instead.' % deprecated)
        ct = mag_client.cluster_templates.create(**args)
        print('Request to create cluster template %s accepted' % parsed_args.name)
        return _show_cluster_template(ct)