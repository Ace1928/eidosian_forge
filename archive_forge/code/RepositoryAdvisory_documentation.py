from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable
import github.AdvisoryVulnerability
import github.NamedUser
from github.AdvisoryBase import AdvisoryBase
from github.AdvisoryCredit import AdvisoryCredit, Credit
from github.AdvisoryCreditDetailed import AdvisoryCreditDetailed
from github.GithubObject import Attribute, NotSet, Opt

        Closes the advisory.
        :calls: `PATCH /repos/{owner}/{repo}/security-advisories/:advisory_id <https://docs.github.com/en/rest/security-advisories/repository-advisories>`
        