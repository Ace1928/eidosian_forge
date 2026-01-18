from __future__ import annotations
from typing import TYPE_CHECKING
from ..core import BaseDomain, DomainIdentityMixin
class ManagedCertificateStatus(BaseDomain):
    """ManagedCertificateStatus Domain

    :param issuance: str
           Status of the issuance process of the Certificate
    :param renewal: str
           Status of the renewal process of the Certificate
    :param error: ManagedCertificateError
          If issuance or renewal reports failure, this property contains information about what happened
    """

    def __init__(self, issuance: str | None=None, renewal: str | None=None, error: ManagedCertificateError | None=None):
        self.issuance = issuance
        self.renewal = renewal
        self.error = error