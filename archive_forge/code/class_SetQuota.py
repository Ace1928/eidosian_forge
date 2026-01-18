import argparse
import itertools
import logging
import sys
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class SetQuota(common.NetDetectionMixin, command.Command):
    _description = _('Set quotas for project or class')

    def _build_options_list(self):
        help_fmt = _('New value for the %s quota')
        rets = [(k, v, help_fmt % v) for k, v in itertools.chain(COMPUTE_QUOTAS.items(), VOLUME_QUOTAS.items())]
        if self.is_docs_build:
            inv_compute = set(NOVA_NETWORK_QUOTAS.values())
            for k, v in NETWORK_QUOTAS.items():
                _help = help_fmt % v
                if v not in inv_compute:
                    _help = self.enhance_help_neutron(_help)
                rets.append((k, v, _help))
        elif self.is_neutron:
            rets.extend([(k, v, help_fmt % v) for k, v in NETWORK_QUOTAS.items()])
        elif self.is_nova_network:
            rets.extend([(k, v, help_fmt % v) for k, v in NOVA_NETWORK_QUOTAS.items()])
        return rets

    def get_parser(self, prog_name):
        parser = super(SetQuota, self).get_parser(prog_name)
        parser.add_argument('project', metavar='<project/class>', help=_('Set quotas for this project or class (name or ID)'))
        parser.add_argument('--class', dest='quota_class', action='store_true', default=False, help=_('**Deprecated** Set quotas for <class>. Deprecated as quota classes were never fully implemented and only the default class is supported. (compute and volume only)'))
        for k, v, h in self._build_options_list():
            parser.add_argument('--%s' % v, metavar='<%s>' % v, dest=k, type=int, help=h)
        parser.add_argument('--volume-type', metavar='<volume-type>', help=_('Set quotas for a specific <volume-type>'))
        force_group = parser.add_mutually_exclusive_group()
        force_group.add_argument('--force', action='store_true', dest='force', default=None, help=_('Force quota update (only supported by compute and network) (default for network)'))
        force_group.add_argument('--no-force', action='store_false', dest='force', default=None, help=_('Do not force quota update (only supported by compute and network) (default for compute)'))
        force_group.add_argument('--check-limit', action='store_false', dest='force', default=None, help=argparse.SUPPRESS)
        return parser

    def take_action(self, parsed_args):
        if parsed_args.quota_class:
            msg = _("The '--class' option has been deprecated. Quota classes were never fully implemented and the compute and volume services only support a single 'default' quota class while the network service does not support quota classes at all. Please use 'openstack quota show --default' instead.")
            self.log.warning(msg)
        identity_client = self.app.client_manager.identity
        compute_client = self.app.client_manager.compute
        volume_client = self.app.client_manager.volume
        compute_kwargs = {}
        for k, v in COMPUTE_QUOTAS.items():
            value = getattr(parsed_args, k, None)
            if value is not None:
                compute_kwargs[k] = value
        if parsed_args.force is not None:
            compute_kwargs['force'] = parsed_args.force
        volume_kwargs = {}
        for k, v in VOLUME_QUOTAS.items():
            value = getattr(parsed_args, k, None)
            if value is not None:
                if parsed_args.volume_type and k in IMPACT_VOLUME_TYPE_QUOTAS:
                    k = k + '_%s' % parsed_args.volume_type
                volume_kwargs[k] = value
        network_kwargs = {}
        if parsed_args.force is True:
            network_kwargs['force'] = True
        elif parsed_args.force is False:
            network_kwargs['check_limit'] = True
        else:
            msg = _("This command currently defaults to '--force' when modifying network quotas. This behavior will change in a future release. Consider explicitly providing '--force' or '--no-force' options to avoid changes in behavior.")
            self.log.warning(msg)
        if self.app.client_manager.is_network_endpoint_enabled():
            for k, v in NETWORK_QUOTAS.items():
                value = getattr(parsed_args, k, None)
                if value is not None:
                    network_kwargs[k] = value
        else:
            for k, v in NOVA_NETWORK_QUOTAS.items():
                value = getattr(parsed_args, k, None)
                if value is not None:
                    compute_kwargs[k] = value
        if parsed_args.quota_class:
            if compute_kwargs:
                compute_client.quota_classes.update(parsed_args.project, **compute_kwargs)
            if volume_kwargs:
                volume_client.quota_classes.update(parsed_args.project, **volume_kwargs)
            if network_kwargs:
                sys.stderr.write('Network quotas are ignored since quota classes are not supported.')
        else:
            project = utils.find_resource(identity_client.projects, parsed_args.project).id
            if compute_kwargs:
                compute_client.quotas.update(project, **compute_kwargs)
            if volume_kwargs:
                volume_client.quotas.update(project, **volume_kwargs)
            if network_kwargs and self.app.client_manager.is_network_endpoint_enabled():
                network_client = self.app.client_manager.network
                network_client.update_quota(project, **network_kwargs)