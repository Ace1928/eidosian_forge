from enum import Enum, IntEnum
from gitlab._version import __title__, __version__
class DetailedMergeStatus(GitlabEnum):
    BLOCKED_STATUS: str = 'blocked_status'
    BROKEN_STATUS: str = 'broken_status'
    CHECKING: str = 'checking'
    UNCHECKED: str = 'unchecked'
    CI_MUST_PASS: str = 'ci_must_pass'
    CI_STILL_RUNNING: str = 'ci_still_running'
    DISCUSSIONS_NOT_RESOLVED: str = 'discussions_not_resolved'
    DRAFT_STATUS: str = 'draft_status'
    EXTERNAL_STATUS_CHECKS: str = 'external_status_checks'
    MERGEABLE: str = 'mergeable'
    NOT_APPROVED: str = 'not_approved'
    NOT_OPEN: str = 'not_open'
    POLICIES_DENIED: str = 'policies_denied'