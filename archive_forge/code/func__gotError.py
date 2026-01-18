import SOAPpy
from twisted.internet import defer
from twisted.web import client, resource, server
def _gotError(self, failure, request, methodName):
    e = failure.value
    if isinstance(e, SOAPpy.faultType):
        fault = e
    else:
        fault = SOAPpy.faultType('%s:Server' % SOAPpy.NS.ENV_T, 'Method %s failed.' % methodName)
    response = SOAPpy.buildSOAP(fault, encoding=self.encoding)
    self._sendResponse(request, response, status=500)