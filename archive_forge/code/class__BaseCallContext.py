import abc
import logging
from oslo_config import cfg
from oslo_messaging._drivers import base as driver_base
from oslo_messaging import _metrics as metrics
from oslo_messaging import _utils as utils
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging import transport as msg_transport
class _BaseCallContext(object, metaclass=abc.ABCMeta):
    _marker = object()

    def __init__(self, transport, target, serializer, timeout=None, version_cap=None, retry=None, call_monitor_timeout=None, transport_options=None):
        self.conf = transport.conf
        self.transport = transport
        self.target = target
        self.serializer = serializer
        self.timeout = timeout
        self.call_monitor_timeout = call_monitor_timeout
        self.retry = retry
        self.version_cap = version_cap
        self.transport_options = transport_options
        super(_BaseCallContext, self).__init__()

    def _make_message(self, ctxt, method, args):
        msg = dict(method=method)
        msg['args'] = dict()
        for argname, arg in args.items():
            msg['args'][argname] = self.serializer.serialize_entity(ctxt, arg)
        if self.target.namespace is not None:
            msg['namespace'] = self.target.namespace
        if self.target.version is not None:
            msg['version'] = self.target.version
        return msg

    def _check_version_cap(self, version):
        if not utils.version_is_compatible(self.version_cap, version):
            raise RPCVersionCapError(version=version, version_cap=self.version_cap)

    def can_send_version(self, version=_marker):
        """Check to see if a version is compatible with the version cap."""
        version = self.target.version if version is self._marker else version
        return utils.version_is_compatible(self.version_cap, version)

    @classmethod
    def _check_version(cls, version):
        if version is not cls._marker:
            try:
                utils.version_is_compatible(version, version)
            except (IndexError, ValueError):
                raise exceptions.MessagingException('Version must contain a major and minor integer. Got %s' % version)

    def cast(self, ctxt, method, **kwargs):
        """Invoke a method and return immediately. See RPCClient.cast()."""
        msg = self._make_message(ctxt, method, kwargs)
        msg_ctxt = self.serializer.serialize_context(ctxt)
        self._check_version_cap(msg.get('version'))
        with metrics.get_collector(self.conf, 'rpc_client', target=self.target, method=method, call_type='cast') as metrics_collector:
            try:
                self.transport._send(self.target, msg_ctxt, msg, retry=self.retry, transport_options=self.transport_options)
            except driver_base.TransportDriverError as ex:
                self._metrics_api.rpc_client_exception_total(self.target, method, 'cast', ex.__class__.__name__)
                raise ClientSendError(self.target, ex)
            except Exception as ex:
                if self.conf.oslo_messaging_metrics.metrics_enabled:
                    metrics_collector.rpc_client_exception_total(self.target, method, 'cast', ex.__class__.__name__)
                raise

    def call(self, ctxt, method, **kwargs):
        """Invoke a method and wait for a reply. See RPCClient.call()."""
        if self.target.fanout:
            raise exceptions.InvalidTarget('A call cannot be used with fanout', self.target)
        msg = self._make_message(ctxt, method, kwargs)
        msg_ctxt = self.serializer.serialize_context(ctxt)
        timeout = self.timeout
        if self.timeout is None:
            timeout = self.conf.rpc_response_timeout
        cm_timeout = self.call_monitor_timeout
        self._check_version_cap(msg.get('version'))
        with metrics.get_collector(self.conf, 'rpc_client', target=self.target, method=method, call_type='call') as metrics_collector:
            try:
                result = self.transport._send(self.target, msg_ctxt, msg, wait_for_reply=True, timeout=timeout, call_monitor_timeout=cm_timeout, retry=self.retry, transport_options=self.transport_options)
            except driver_base.TransportDriverError as ex:
                self._metrics_api.rpc_client_exception_total(self.target, method, 'call', ex.__class__.__name__)
                raise ClientSendError(self.target, ex)
            except Exception as ex:
                if self.conf.oslo_messaging_metrics.metrics_enabled:
                    metrics_collector.rpc_client_exception_total(self.target, method, 'call', ex.__class__.__name__)
                raise
            return self.serializer.deserialize_entity(ctxt, result)

    @abc.abstractmethod
    def prepare(self, exchange=_marker, topic=_marker, namespace=_marker, version=_marker, server=_marker, fanout=_marker, timeout=_marker, version_cap=_marker, retry=_marker, call_monitor_timeout=_marker):
        """Prepare a method invocation context. See RPCClient.prepare()."""