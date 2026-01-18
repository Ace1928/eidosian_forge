from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import (
class BoundCertificate(BoundModelBase, Certificate):
    _client: CertificatesClient
    model = Certificate

    def __init__(self, client: CertificatesClient, data: dict, complete: bool=True):
        status = data.get('status')
        if status is not None:
            error_data = status.get('error')
            error = None
            if error_data:
                error = ManagedCertificateError(code=error_data['code'], message=error_data['message'])
            data['status'] = ManagedCertificateStatus(issuance=status['issuance'], renewal=status['renewal'], error=error)
        super().__init__(client, data, complete)

    def get_actions_list(self, status: list[str] | None=None, sort: list[str] | None=None, page: int | None=None, per_page: int | None=None) -> ActionsPageResult:
        """Returns all action objects for a Certificate.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :param page: int (optional)
               Specifies the page to fetch
        :param per_page: int (optional)
               Specifies how many results are returned by page
        :return: (List[:class:`BoundAction <hcloud.actions.client.BoundAction>`], :class:`Meta <hcloud.core.domain.Meta>`)
        """
        return self._client.get_actions_list(self, status, sort, page, per_page)

    def get_actions(self, status: list[str] | None=None, sort: list[str] | None=None) -> list[BoundAction]:
        """Returns all action objects for a Certificate.

        :param status: List[str] (optional)
               Response will have only actions with specified statuses. Choices: `running` `success` `error`
        :param sort: List[str] (optional)
               Specify how the results are sorted. Choices: `id` `id:asc` `id:desc` `command` `command:asc` `command:desc` `status` `status:asc` `status:desc` `progress` `progress:asc` `progress:desc` `started` `started:asc` `started:desc` `finished` `finished:asc` `finished:desc`
        :return: List[:class:`BoundAction <hcloud.actions.client.BoundAction>`]
        """
        return self._client.get_actions(self, status, sort)

    def update(self, name: str | None=None, labels: dict[str, str] | None=None) -> BoundCertificate:
        """Updates an certificate. You can update an certificate name and the certificate labels.

        :param name: str (optional)
               New name to set
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :return: :class:`BoundCertificate <hcloud.certificates.client.BoundCertificate>`
        """
        return self._client.update(self, name, labels)

    def delete(self) -> bool:
        """Deletes a certificate.
        :return: boolean
        """
        return self._client.delete(self)

    def retry_issuance(self) -> BoundAction:
        """Retry a failed Certificate issuance or renewal.
        :return: BoundAction
        """
        return self._client.retry_issuance(self)