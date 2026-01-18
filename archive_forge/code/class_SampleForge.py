import os
from typing import List
from .. import forge as _mod_forge
from .. import registry, tests, urlutils
from ..forge import (Forge, MergeProposal, UnsupportedForge, determine_title,
class SampleForge(Forge):
    _locations: List[str] = []

    @classmethod
    def _add_location(cls, url):
        cls._locations.append(url)

    @classmethod
    def probe_from_url(cls, url, possible_transports=None):
        for b in cls._locations:
            if url.startswith(b):
                return cls()
        raise UnsupportedForge(url)

    def hosts(self, branch):
        for b in self._locations:
            if branch.user_url.startswith(b):
                return True
        return False

    @classmethod
    def iter_instances(cls):
        return iter([cls()])

    def get_proposal_by_url(self, url):
        for b in self._locations:
            if url.startswith(b):
                return MergeProposal()
        raise UnsupportedForge(url)