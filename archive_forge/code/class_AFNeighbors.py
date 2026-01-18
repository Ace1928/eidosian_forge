from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
class AFNeighbors(CliProvider):

    def render(self, config=None, nbr_list=None):
        commands = list()
        if not nbr_list:
            return
        for item in nbr_list:
            neighbor_commands = list()
            for key, value in iteritems(item):
                if value is not None:
                    meth = getattr(self, '_render_%s' % key, None)
                    if meth:
                        resp = meth(item, config)
                        if resp:
                            neighbor_commands.extend(to_list(resp))
            commands.extend(neighbor_commands)
        return commands

    def _render_route_reflector_client(self, item, config=None):
        cmd = 'neighbor %s route-reflector-client' % item['neighbor']
        if item['route_reflector_client'] is False:
            if not config or cmd in config:
                cmd = 'no %s' % cmd
                return cmd
        elif not config or cmd not in config:
            return cmd

    def _render_route_server_client(self, item, config=None):
        cmd = 'neighbor %s route-server-client' % item['neighbor']
        if item['route_server_client'] is False:
            if not config or cmd in config:
                cmd = 'no %s' % cmd
                return cmd
        elif not config or cmd not in config:
            return cmd

    def _render_remove_private_as(self, item, config=None):
        cmd = 'neighbor %s remove-private-AS' % item['neighbor']
        if item['remove_private_as'] is False:
            if not config or cmd in config:
                cmd = 'no %s' % cmd
                return cmd
        elif not config or cmd not in config:
            return cmd

    def _render_next_hop_self(self, item, config=None):
        cmd = 'neighbor %s activate' % item['neighbor']
        if item['next_hop_self'] is False:
            if not config or cmd in config:
                cmd = 'no %s' % cmd
                return cmd
        elif not config or cmd not in config:
            return cmd

    def _render_activate(self, item, config=None):
        cmd = 'neighbor %s activate' % item['neighbor']
        if item['activate'] is False:
            if not config or cmd in config:
                cmd = 'no %s' % cmd
                return cmd
        elif not config or cmd not in config:
            return cmd

    def _render_maximum_prefix(self, item, config=None):
        cmd = 'neighbor %s maximum-prefix %s' % (item['neighbor'], item['maximum_prefix'])
        if not config or cmd not in config:
            return cmd