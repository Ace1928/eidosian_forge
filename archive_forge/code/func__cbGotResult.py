import SOAPpy
from twisted.internet import defer
from twisted.web import client, resource, server
def _cbGotResult(self, result):
    result = SOAPpy.parseSOAPRPC(result)
    if hasattr(result, 'Result'):
        return result.Result
    elif len(result) == 1:
        return result[0]
    else:
        return result