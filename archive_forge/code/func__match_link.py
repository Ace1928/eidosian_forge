from typing import FrozenSet, Iterable, Optional, Tuple, Union
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName
from pip._vendor.packaging.version import LegacyVersion, Version
from pip._internal.models.link import Link, links_equivalent
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.hashes import Hashes
def _match_link(link: Link, candidate: 'Candidate') -> bool:
    if candidate.source_link:
        return links_equivalent(link, candidate.source_link)
    return False