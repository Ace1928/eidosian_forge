from requests.exceptions import RequestException
class KerberosExchangeError(RequestException):
    """Kerberos Exchange Failed Error"""