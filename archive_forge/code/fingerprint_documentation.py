from requests.adapters import HTTPAdapter
from .._compat import poolmanager

    A HTTPS Adapter for Python Requests that verifies certificate fingerprints,
    instead of certificate hostnames.

    Example usage:

    .. code-block:: python

        import requests
        import ssl
        from requests_toolbelt.adapters.fingerprint import FingerprintAdapter

        twitter_fingerprint = '...'
        s = requests.Session()
        s.mount(
            'https://twitter.com',
            FingerprintAdapter(twitter_fingerprint)
        )

    The fingerprint should be provided as a hexadecimal string, optionally
    containing colons.
    