import abc
import logging
from oslo_messaging.target import Target
class LegacyAddresser(Addresser):
    """Legacy addresses are in the following format:

    multicast: '$broadcast_prefix[.$vhost].$exchange.$topic.all'
    unicast: '$server_prefix[.$vhost].$exchange.$topic.$server'
    anycast: '$group_prefix[.$vhost].$exchange.$topic'

    Legacy addresses do not distinguish RPC traffic from Notification traffic
    """

    def __init__(self, default_exchange, server_prefix, broadcast_prefix, group_prefix, vhost):
        super(LegacyAddresser, self).__init__(default_exchange)
        self._server_prefix = server_prefix
        self._broadcast_prefix = broadcast_prefix
        self._group_prefix = group_prefix
        self._vhost = vhost

    def multicast_address(self, target, service):
        return self._concat('.', [self._broadcast_prefix, self._vhost, target.exchange or self._default_exchange, target.topic, 'all'])

    def unicast_address(self, target, service=SERVICE_RPC):
        return self._concat('.', [self._server_prefix, self._vhost, target.exchange or self._default_exchange, target.topic, target.server])

    def anycast_address(self, target, service=SERVICE_RPC):
        return self._concat('.', [self._group_prefix, self._vhost, target.exchange or self._default_exchange, target.topic])

    def _is_multicast(self, address):
        return address.startswith(self._broadcast_prefix)

    def _is_unicast(self, address):
        return address.startswith(self._server_prefix)

    def _is_anycast(self, address):
        return address.startswith(self._group_prefix)

    def _is_service(self, address, service):
        return True