import zmq
from zmq.devices.proxydevice import ProcessProxy, Proxy, ThreadProxy
def _setup_sockets(self):
    ins, outs, mons = super()._setup_sockets()
    ctx = self._context
    ctrls = ctx.socket(self.ctrl_type)
    self._sockets.append(ctrls)
    for opt, value in self._ctrl_sockopts:
        ctrls.setsockopt(opt, value)
    for iface in self._ctrl_binds:
        ctrls.bind(iface)
    for iface in self._ctrl_connects:
        ctrls.connect(iface)
    return (ins, outs, mons, ctrls)