import dataclasses
import enum
import json
import sys
from typing import Any, Mapping, Optional
from apitools.base.py import encoding
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages
@dataclasses.dataclass(repr=False)
class AgentsRule(object):
    """An Ops agents rule contains package state, and version.

    Attr:
      version: agent version, e.g. 'latest', '2.52.1'.
      package_state: desired state for the package.
    """

    class PackageState(*_StrEnum):
        INSTALLED = 'installed'
        REMOVED = 'removed'
    version: Optional[str]
    package_state: PackageState = PackageState.INSTALLED

    def __repr__(self) -> str:
        """JSON single line format string."""
        return self.ToJson()

    def ToJson(self) -> str:
        """JSON single line format string."""
        return json.dumps(self.__dict__, separators=(',', ':'), default=str, sort_keys=True)