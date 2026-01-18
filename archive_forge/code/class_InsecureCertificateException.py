from typing import Optional
from typing import Sequence
class InsecureCertificateException(WebDriverException):
    """Navigation caused the user agent to hit a certificate warning, which is
    usually the result of an expired or invalid TLS certificate."""