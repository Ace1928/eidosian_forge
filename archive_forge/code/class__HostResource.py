from twisted.python import roots
from twisted.web import pages, resource
class _HostResource(resource.Resource):

    def getChild(self, path, request):
        if b':' in path:
            host, port = path.split(b':', 1)
            port = int(port)
        else:
            host, port = (path, 80)
        request.setHost(host, port)
        prefixLen = 3 + request.isSecure() + 4 + len(path) + len(request.prepath[-3])
        request.path = b'/' + b'/'.join(request.postpath)
        request.uri = request.uri[prefixLen:]
        del request.prepath[:3]
        return request.site.getResourceFor(request)