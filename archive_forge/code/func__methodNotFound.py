import SOAPpy
from twisted.internet import defer
from twisted.web import client, resource, server
def _methodNotFound(self, request, methodName):
    response = SOAPpy.buildSOAP(SOAPpy.faultType('%s:Client' % SOAPpy.NS.ENV_T, 'Method %s not found' % methodName), encoding=self.encoding)
    self._sendResponse(request, response, status=500)