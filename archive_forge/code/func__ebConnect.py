from twisted.conch.client import direct
def _ebConnect(f, useConnects, host, port, options, vhk, uao):
    if not useConnects:
        return f
    connectType = useConnects.pop(0)
    f = connectTypes[connectType]
    d = f(host, port, options, vhk, uao)
    d.addErrback(_ebConnect, useConnects, host, port, options, vhk, uao)
    return d