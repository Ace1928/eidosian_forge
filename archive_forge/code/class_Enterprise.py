import urllib.parse
from typing import Any, Dict
from github.EnterpriseConsumedLicenses import EnterpriseConsumedLicenses
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester
class Enterprise(NonCompletableGithubObject):
    """
    This class represents Enterprises. Such objects do not exist in the Github API, so this class merely collects all endpoints the start with /enterprises/{enterprise}/. See methods below for specific endpoints and docs.
    https://docs.github.com/en/enterprise-cloud@latest/rest/enterprise-admin?apiVersion=2022-11-28
    """

    def __init__(self, requester: Requester, enterprise: str):
        enterprise = urllib.parse.quote(enterprise)
        super().__init__(requester, {}, {'enterprise': enterprise, 'url': f'/enterprises/{enterprise}'}, True)

    def _initAttributes(self) -> None:
        self._enterprise: Attribute[str] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'enterprise': self._enterprise.value})

    @property
    def enterprise(self) -> str:
        return self._enterprise.value

    @property
    def url(self) -> str:
        return self._url.value

    def get_consumed_licenses(self) -> EnterpriseConsumedLicenses:
        """
        :calls: `GET /enterprises/{enterprise}/consumed-licenses <https://docs.github.com/en/enterprise-cloud@latest/rest/enterprise-admin/license#list-enterprise-consumed-licenses>`_
        """
        headers, data = self._requester.requestJsonAndCheck('GET', self.url + '/consumed-licenses')
        if 'url' not in data:
            data['url'] = self.url + '/consumed-licenses'
        return EnterpriseConsumedLicenses(self._requester, headers, data, completed=True)

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'enterprise' in attributes:
            self._enterprise = self._makeStringAttribute(attributes['enterprise'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])