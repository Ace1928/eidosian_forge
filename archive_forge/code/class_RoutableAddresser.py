import abc
import logging
from oslo_messaging.target import Target
class RoutableAddresser(Addresser):
    """Routable addresses have different formats based their use.  It starts
    with a prefix that is determined by the type of traffic (RPC or
    Notifications).  The prefix is followed by a description of messaging
    delivery semantics. The delivery may be one of: 'multicast', 'unicast', or
    'anycast'. The delivery semantics are followed by information pulled from
    the Target.  The template is:

    $prefix/$semantics[/$vhost]/$exchange/$topic[/$server]

    Examples based on the default prefix and semantic values:

    rpc-unicast: "openstack.org/om/rpc/unicast/my-exchange/my-topic/my-server"
    notify-anycast: "openstack.org/om/notify/anycast/my-vhost/exchange/topic"
    """

    def __init__(self, default_exchange, rpc_exchange, rpc_prefix, notify_exchange, notify_prefix, unicast_tag, multicast_tag, anycast_tag, vhost):
        super(RoutableAddresser, self).__init__(default_exchange)
        if not self._default_exchange:
            self._default_exchange = 'openstack'
        self._vhost = vhost
        _rpc = rpc_prefix + '/'
        self._rpc_prefix = _rpc
        self._rpc_unicast = _rpc + unicast_tag
        self._rpc_multicast = _rpc + multicast_tag
        self._rpc_anycast = _rpc + anycast_tag
        _notify = notify_prefix + '/'
        self._notify_prefix = _notify
        self._notify_unicast = _notify + unicast_tag
        self._notify_multicast = _notify + multicast_tag
        self._notify_anycast = _notify + anycast_tag
        self._exchange = [rpc_exchange or self._default_exchange or 'rpc', notify_exchange or self._default_exchange or 'notify']

    def multicast_address(self, target, service=SERVICE_RPC):
        if service == SERVICE_RPC:
            prefix = self._rpc_multicast
        else:
            prefix = self._notify_multicast
        return self._concat('/', [prefix, self._vhost, target.exchange or self._exchange[service], target.topic])

    def unicast_address(self, target, service=SERVICE_RPC):
        if service == SERVICE_RPC:
            prefix = self._rpc_unicast
        else:
            prefix = self._notify_unicast
        return self._concat('/', [prefix, self._vhost, target.exchange or self._exchange[service], target.topic, target.server])

    def anycast_address(self, target, service=SERVICE_RPC):
        if service == SERVICE_RPC:
            prefix = self._rpc_anycast
        else:
            prefix = self._notify_anycast
        return self._concat('/', [prefix, self._vhost, target.exchange or self._exchange[service], target.topic])

    def _is_multicast(self, address):
        return address.startswith(self._rpc_multicast) or address.startswith(self._notify_multicast)

    def _is_unicast(self, address):
        return address.startswith(self._rpc_unicast) or address.startswith(self._notify_unicast)

    def _is_anycast(self, address):
        return address.startswith(self._rpc_anycast) or address.startswith(self._notify_anycast)

    def _is_service(self, address, service):
        return address.startswith(self._rpc_prefix if service == SERVICE_RPC else self._notify_prefix)