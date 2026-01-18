from __future__ import annotations
from datetime import datetime
from typing import Any
from github.CVSS import CVSS
from github.CWE import CWE
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet

    This class represents a the shared attributes between GlobalAdvisory, RepositoryAdvisory and DependabotAdvisory
    https://docs.github.com/en/rest/security-advisories/global-advisories
    https://docs.github.com/en/rest/security-advisories/repository-advisories
    https://docs.github.com/en/rest/dependabot/alerts
    