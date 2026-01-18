from abc import ABCMeta
from abc import abstractmethod
import logging
import sys
import threading
from oslo_config import cfg
from oslo_utils import eventletutils
from oslo_messaging import _utils as utils
from oslo_messaging import dispatcher
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import server as msg_server
from oslo_messaging import target as msg_target
class RPCDispatcher(dispatcher.DispatcherBase):
    """A message dispatcher which understands RPC messages.

    A MessageHandlingServer is constructed by passing a callable dispatcher
    which is invoked with context and message dictionaries each time a message
    is received.

    RPCDispatcher is one such dispatcher which understands the format of RPC
    messages. The dispatcher looks at the namespace, version and method values
    in the message and matches those against a list of available endpoints.

    Endpoints may have a target attribute describing the namespace and version
    of the methods exposed by that object.

    The RPCDispatcher may have an access_policy attribute which determines
    which of the endpoint methods are to be dispatched.
    The default access_policy dispatches all public methods
    on an endpoint object.


    """

    def __init__(self, endpoints, serializer, access_policy=None):
        """Construct a rpc server dispatcher.

        :param endpoints: list of endpoint objects for dispatching to
        :param serializer: optional message serializer
        """
        cfg.CONF.register_opts(_dispatcher_opts)
        oslo_rpc_server_ping = None
        for ep in endpoints:
            target = getattr(ep, 'target', None)
            if target and (not isinstance(target, msg_target.Target)):
                errmsg = "'target' is a reserved Endpoint attribute used" + ' for namespace and version filtering.  It must' + ' be of type oslo_messaging.Target. Do not' + " define an Endpoint method named 'target'"
                raise TypeError('%s: endpoint=%s' % (errmsg, ep))
            oslo_rpc_server_ping = getattr(ep, 'oslo_rpc_server_ping', None)
            if oslo_rpc_server_ping:
                errmsg = "'oslo_rpc_server_ping' is a reserved Endpoint" + ' attribute which can be use to ping the' + ' endpoint. Please avoid using any oslo_* ' + ' naming.'
                LOG.warning('%s (endpoint=%s)' % (errmsg, ep))
        self.endpoints = endpoints
        if cfg.CONF.rpc_ping_enabled:
            if oslo_rpc_server_ping:
                LOG.warning('rpc_ping_enabled=True in config but oslo_rpc_server_ping is already declared in an other Endpoint. Not enabling rpc_ping Endpoint.')
            else:
                self.endpoints.append(PingEndpoint())
        self.serializer = serializer or msg_serializer.NoOpSerializer()
        self._default_target = msg_target.Target()
        if access_policy is not None:
            if issubclass(access_policy, RPCAccessPolicyBase):
                self.access_policy = access_policy()
            else:
                raise TypeError('access_policy must be a subclass of RPCAccessPolicyBase')
        else:
            self.access_policy = DefaultRPCAccessPolicy()

    @staticmethod
    def _is_namespace(target, namespace):
        return namespace in target.accepted_namespaces

    @staticmethod
    def _is_compatible(target, version):
        endpoint_version = target.version or '1.0'
        return utils.version_is_compatible(endpoint_version, version)

    def _do_dispatch(self, endpoint, method, ctxt, args):
        ctxt = self.serializer.deserialize_context(ctxt)
        new_args = dict()
        for argname, arg in args.items():
            new_args[argname] = self.serializer.deserialize_entity(ctxt, arg)
        func = getattr(endpoint, method)
        result = func(ctxt, **new_args)
        return self.serializer.serialize_entity(ctxt, result)

    def _watchdog(self, event, incoming):
        try:
            client_timeout = int(incoming.client_timeout)
            cm_heartbeat_interval = client_timeout / 2
        except ValueError:
            client_timeout = cm_heartbeat_interval = 0
        if cm_heartbeat_interval < 1:
            LOG.warning('Client provided an invalid timeout value of %r' % incoming.client_timeout)
            return
        while not event.wait(cm_heartbeat_interval):
            LOG.debug('Sending call-monitor heartbeat for active call to %(method)s (interval=%(interval)i)' % {'method': incoming.message.get('method'), 'interval': cm_heartbeat_interval})
            try:
                incoming.heartbeat()
            except Exception as exc:
                LOG.debug('Call-monitor heartbeat failed: %(exc)s' % {'exc': exc})
                break

    def dispatch(self, incoming):
        """Dispatch an RPC message to the appropriate endpoint method.

        :param incoming: incoming message
        :type incoming: IncomingMessage
        :raises: NoSuchMethod, UnsupportedVersion
        """
        message = incoming.message
        ctxt = incoming.ctxt
        method = message.get('method')
        args = message.get('args', {})
        namespace = message.get('namespace')
        version = message.get('version', '1.0')
        completion_event = eventletutils.Event()
        watchdog_thread = threading.Thread(target=self._watchdog, args=(completion_event, incoming))
        if incoming.client_timeout:
            watchdog_thread.start()
        found_compatible = False
        for endpoint in self.endpoints:
            target = getattr(endpoint, 'target', None)
            if not target:
                target = self._default_target
            if not (self._is_namespace(target, namespace) and self._is_compatible(target, version)):
                continue
            if hasattr(endpoint, method):
                if self.access_policy.is_allowed(endpoint, method):
                    try:
                        return self._do_dispatch(endpoint, method, ctxt, args)
                    finally:
                        completion_event.set()
                        if incoming.client_timeout:
                            watchdog_thread.join()
            found_compatible = True
        if found_compatible:
            raise NoSuchMethod(method)
        else:
            raise UnsupportedVersion(version, method=method)