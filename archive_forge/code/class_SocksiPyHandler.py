import ssl
import socks # $ pip install PySocks
class SocksiPyHandler(urllib2.HTTPHandler, urllib2.HTTPSHandler):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kw = kwargs
        urllib2.HTTPHandler.__init__(self)

    def http_open(self, req):

        def build(host, port=None, timeout=0, **kwargs):
            kw = merge_dict(self.kw, kwargs)
            conn = SocksiPyConnection(*self.args, host=host, port=port, timeout=timeout, **kw)
            return conn
        return self.do_open(build, req)

    def https_open(self, req):

        def build(host, port=None, timeout=0, **kwargs):
            kw = merge_dict(self.kw, kwargs)
            conn = SocksiPyConnectionS(*self.args, host=host, port=port, timeout=timeout, **kw)
            return conn
        return self.do_open(build, req)