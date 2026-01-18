from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApprovalStateValueValuesEnum(_messages.Enum):
    """Output only. Approval state of the blueprint (DRAFT, PROPOSED,
    APPROVED)

    Values:
      APPROVAL_STATE_UNSPECIFIED: Unspecified state.
      DRAFT: A blueprint starts in DRAFT state once it is created. All edits
        are made to the blueprint in DRAFT state.
      PROPOSED: When the edits are ready for review, blueprint can be proposed
        and moves to PROPOSED state. Edits cannot be made to a blueprint in
        PROPOSED state.
      APPROVED: When a proposed blueprint is approved, it moves to APPROVED
        state. A new revision is committed. The latest committed revision can
        be used to create a deployment on Orchestration or Workload Cluster.
        Edits to an APPROVED blueprint changes its state back to DRAFT. The
        last committed revision of a blueprint represents its latest APPROVED
        state.
    """
    APPROVAL_STATE_UNSPECIFIED = 0
    DRAFT = 1
    PROPOSED = 2
    APPROVED = 3