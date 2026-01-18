import copy
import os
import sys
from io import BytesIO
from xml.dom.minidom import getDOMImplementation
from twisted.internet import address, reactor
from twisted.logger import Logger
from twisted.persisted import styles
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.web import http, resource, server, static, util
from twisted.web.http_headers import Headers
class ResourcePublisher(pb.Root, styles.Versioned):
    """
    L{ResourcePublisher} exposes a remote API which can be used to respond
    to request.

    @ivar site: The site which will be used for resource lookup.
    @type site: L{twisted.web.server.Site}
    """
    _log = Logger()

    def __init__(self, site):
        self.site = site
    persistenceVersion = 2

    def upgradeToVersion2(self):
        self.application.authorizer.removeIdentity('web')
        del self.application.services[self.serviceName]
        del self.serviceName
        del self.application
        del self.perspectiveName

    def getPerspectiveNamed(self, name):
        return self

    def remote_request(self, request):
        """
        Look up the resource for the given request and render it.
        """
        res = self.site.getResourceFor(request)
        self._log.info(request)
        result = res.render(request)
        if result is not server.NOT_DONE_YET:
            request.write(result)
            request.finish()
        return server.NOT_DONE_YET