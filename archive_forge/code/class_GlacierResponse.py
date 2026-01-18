from boto.compat import json
class GlacierResponse(dict):
    """
    Represents a response from Glacier layer1. It acts as a dictionary
    containing the combined keys received via JSON in the body (if
    supplied) and headers.
    """

    def __init__(self, http_response, response_headers):
        self.http_response = http_response
        self.status = http_response.status
        self[u'RequestId'] = http_response.getheader('x-amzn-requestid')
        if response_headers:
            for header_name, item_name in response_headers:
                self[item_name] = http_response.getheader(header_name)
        if http_response.status != 204:
            if http_response.getheader('Content-Type') == 'application/json':
                body = json.loads(http_response.read().decode('utf-8'))
                self.update(body)
        size = http_response.getheader('Content-Length', None)
        if size is not None:
            self.size = size

    def read(self, amt=None):
        """Reads and returns the response body, or up to the next amt bytes."""
        return self.http_response.read(amt)