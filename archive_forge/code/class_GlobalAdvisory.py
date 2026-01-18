from __future__ import annotations
from datetime import datetime
from typing import Any
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.AdvisoryVulnerability import AdvisoryVulnerability
from github.GithubObject import Attribute, NotSet
class GlobalAdvisory(AdvisoryBase):
    """
    This class represents a GlobalAdvisory.
    https://docs.github.com/en/rest/security-advisories/global-advisories
    """

    def _initAttributes(self) -> None:
        self._credits: Attribute[list[AdvisoryCreditDetailed]] = NotSet
        self._github_reviewed_at: Attribute[datetime] = NotSet
        self._nvd_published_at: Attribute[datetime] = NotSet
        self._references: Attribute[list[str]] = NotSet
        self._repository_advisory_url: Attribute[str] = NotSet
        self._source_code_location: Attribute[str] = NotSet
        self._type: Attribute[str] = NotSet
        self._vulnerabilities: Attribute[list[AdvisoryVulnerability]] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'ghsa_id': self.ghsa_id, 'summary': self.summary})

    @property
    def credits(self) -> list[AdvisoryCreditDetailed]:
        return self._credits.value

    @property
    def github_reviewed_at(self) -> datetime:
        return self._github_reviewed_at.value

    @property
    def nvd_published_at(self) -> datetime:
        return self._nvd_published_at.value

    @property
    def references(self) -> list[str]:
        return self._references.value

    @property
    def repository_advisory_url(self) -> str:
        return self._repository_advisory_url.value

    @property
    def source_code_location(self) -> str:
        return self._source_code_location.value

    @property
    def type(self) -> str:
        return self._type.value

    @property
    def vulnerabilities(self) -> list[AdvisoryVulnerability]:
        return self._vulnerabilities.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'credits' in attributes:
            self._credits = self._makeListOfClassesAttribute(AdvisoryCreditDetailed, attributes['credits'])
        if 'github_reviewed_at' in attributes:
            assert attributes['github_reviewed_at'] is None or isinstance(attributes['github_reviewed_at'], str), attributes['github_reviewed_at']
            self._github_reviewed_at = self._makeDatetimeAttribute(attributes['github_reviewed_at'])
        if 'nvd_published_at' in attributes:
            assert attributes['nvd_published_at'] is None or isinstance(attributes['nvd_published_at'], str), attributes['nvd_published_at']
            self._nvd_published_at = self._makeDatetimeAttribute(attributes['nvd_published_at'])
        if 'references' in attributes:
            self._references = self._makeListOfStringsAttribute(attributes['references'])
        if 'repository_advisory_url' in attributes:
            self._repository_advisory_url = self._makeStringAttribute(attributes['repository_advisory_url'])
        if 'source_code_location' in attributes:
            self._source_code_location = self._makeStringAttribute(attributes['source_code_location'])
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'vulnerabilities' in attributes:
            self._vulnerabilities = self._makeListOfClassesAttribute(AdvisoryVulnerability, attributes['vulnerabilities'])
        super()._useAttributes(attributes)