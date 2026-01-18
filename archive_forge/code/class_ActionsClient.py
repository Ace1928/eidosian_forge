from __future__ import annotations
import time
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import Action, ActionFailedException, ActionTimeoutException
class ActionsClient(ResourceActionsClient):

    def __init__(self, client: Client):
        super().__init__(client, None)

    def get_list(self, status: list[str] | None=None, sort: list[str] | None=None, page: int | None=None, per_page: int | None=None) -> ActionsPageResult:
        """
        .. deprecated:: 1.28
            Use :func:`client.<resource>.actions.get_list` instead,
            e.g. using :attr:`hcloud.certificates.client.CertificatesClient.actions`.

            `Starting 1 October 2023, it will no longer be available. <https://docs.hetzner.cloud/changelog#2023-07-20-actions-list-endpoint-is-deprecated>`_
        """
        warnings.warn("The 'client.actions.get_list' method is deprecated, please use the 'client.<resource>.actions.get_list' method instead (e.g. 'client.certificates.actions.get_list').", DeprecationWarning, stacklevel=2)
        return super().get_list(status=status, sort=sort, page=page, per_page=per_page)

    def get_all(self, status: list[str] | None=None, sort: list[str] | None=None) -> list[BoundAction]:
        """
        .. deprecated:: 1.28
            Use :func:`client.<resource>.actions.get_all` instead,
            e.g. using :attr:`hcloud.certificates.client.CertificatesClient.actions`.

            `Starting 1 October 2023, it will no longer be available. <https://docs.hetzner.cloud/changelog#2023-07-20-actions-list-endpoint-is-deprecated>`_
        """
        warnings.warn("The 'client.actions.get_all' method is deprecated, please use the 'client.<resource>.actions.get_all' method instead (e.g. 'client.certificates.actions.get_all').", DeprecationWarning, stacklevel=2)
        return super().get_all(status=status, sort=sort)