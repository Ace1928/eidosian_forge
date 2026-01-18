class CertificatesManager(object):
    """Manager for certificates."""

    def __init__(self, client):
        self._client = client

    def get_ca_certificate(self):
        """Get CA certificate.

        :returns: PEM-formatted string.
        :rtype: str

        """
        resp, body = self._client.get('/certificates/ca', authenticated=False)
        return resp.text

    def get_signing_certificate(self):
        """Get signing certificate.

        :returns: PEM-formatted string.
        :rtype: str

        """
        resp, body = self._client.get('/certificates/signing', authenticated=False)
        return resp.text