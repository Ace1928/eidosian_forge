from zmq import PUB
from zmq.devices.monitoredqueue import monitored_queue
from zmq.devices.proxydevice import ProcessProxy, Proxy, ProxyBase, ThreadProxy
class ProcessMonitoredQueue(MonitoredQueueBase, ProcessProxy):
    """Run zmq.monitored_queue in a separate process.

    See MonitoredQueue and Proxy for details.
    """