from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain, DomainIdentityMixin
class ManagedCertificateError(BaseDomain):
    """ManagedCertificateError Domain

    :param code: str
        Error code identifying the error
    :param message:
        Message detailing the error
    """

    def __init__(self, code: str | None=None, message: str | None=None):
        self.code = code
        self.message = message