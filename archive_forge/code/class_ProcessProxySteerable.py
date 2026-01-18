import zmq
from zmq.devices.proxydevice import ProcessProxy, Proxy, ThreadProxy
class ProcessProxySteerable(ProxySteerableBase, ProcessProxy):
    """ProxySteerable in a Process. See ProxySteerable for details."""