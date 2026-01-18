from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._internal.req.constructors import install_req_drop_extras
from pip._internal.req.req_install import InstallRequirement
from .base import Candidate, CandidateLookup, Requirement, format_name
class ExplicitRequirement(Requirement):

    def __init__(self, candidate: Candidate) -> None:
        self.candidate = candidate

    def __str__(self) -> str:
        return str(self.candidate)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.candidate!r})'

    @property
    def project_name(self) -> NormalizedName:
        return self.candidate.project_name

    @property
    def name(self) -> str:
        return self.candidate.name

    def format_for_error(self) -> str:
        return self.candidate.format_for_error()

    def get_candidate_lookup(self) -> CandidateLookup:
        return (self.candidate, None)

    def is_satisfied_by(self, candidate: Candidate) -> bool:
        return candidate == self.candidate