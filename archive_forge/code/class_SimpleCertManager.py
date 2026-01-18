from keystoneclient import base
class SimpleCertManager(object):
    """Manager for the OS-SIMPLE-CERT extension."""

    def __init__(self, client):
        self._client = client
        self.mgr = base.Manager(self._client)

    def get_ca_certificates(self):
        """Get CA certificates.

        :returns: PEM-formatted string.
        :rtype: str

        """
        resp, body = self._client.get('/OS-SIMPLE-CERT/ca', authenticated=False)
        return self.mgr._prepare_return_value(resp, resp.text)

    def get_certificates(self):
        """Get signing certificates.

        :returns: PEM-formatted string.
        :rtype: str

        """
        resp, body = self._client.get('/OS-SIMPLE-CERT/certificates', authenticated=False)
        return self.mgr._prepare_return_value(resp, resp.text)