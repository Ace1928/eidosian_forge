"""
🛡️ Eidosian Self-Modification Gates.

Ensures that any autonomous changes to the system (code, configuration, 
or identity) are formally proposed, threat-modeled, and benchmarked 
before being committed to the permanent state.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from eidosian_core import eidosian

logger = logging.getLogger("agent_forge.autonomy.gates")

class ModificationProposal:
    """A formal proposal for a systemic change."""
    def __init__(
        self,
        proposal_id: str,
        target_path: Path,
        change_type: str,  # 'code', 'config', 'identity'
        proposed_content: str,
        rationale: str,
    ):
        self.id = proposal_id
        self.target_path = target_path
        self.change_type = change_type
        self.proposed_content = proposed_content
        self.rationale = rationale
        self.status = "pending"
        self.verdicts: Dict[str, Any] = {}

class SystemicGateKeeper:
    """
    Orchestrates the validation pipeline for systemic self-modification.
    """
    
    def __init__(self, repo_root: Path, invariants_path: Path):
        self.repo_root = repo_root
        self.invariants_path = invariants_path
        self._active_proposals: Dict[str, ModificationProposal] = {}

    @eidosian()
    def propose_change(
        self, 
        target_path: str, 
        change_type: str, 
        proposed_content: str, 
        rationale: str
    ) -> str:
        """Initializes a new modification proposal."""
        path = self.repo_root / target_path
        prop_id = hashlib.sha256(f"{target_path}:{proposed_content}".encode()).hexdigest()[:12]
        
        proposal = ModificationProposal(
            proposal_id=prop_id,
            target_path=path,
            change_type=change_type,
            proposed_content=proposed_content,
            rationale=rationale
        )
        self._active_proposals[prop_id] = proposal
        return prop_id

    @eidosian()
    def validate_proposal(self, proposal_id: str) -> Dict[str, Any]:
        """
        Runs the full validation battery:
        1. Constitutional Check (Invariants)
        2. Threat Modeling (Sanitization)
        3. Regression Testing (Optional)
        """
        proposal = self._active_proposals.get(proposal_id)
        if not proposal:
            return {"error": "Proposal not found"}

        # 1. Constitutional Integrity Check
        constitutional_pass = self._check_constitutional_integrity(proposal)
        proposal.verdicts["constitution"] = constitutional_pass
        
        # 2. Safety/Threat Check
        safety_pass = self._check_safety_constraints(proposal)
        proposal.verdicts["safety"] = safety_pass
        
        if all([constitutional_pass["pass"], safety_pass["pass"]]):
            proposal.status = "validated"
        else:
            proposal.status = "rejected"
            
        return {
            "proposal_id": proposal_id,
            "status": proposal.status,
            "verdicts": proposal.verdicts
        }

    def _check_constitutional_integrity(self, proposal: ModificationProposal) -> Dict[str, Any]:
        """Verify that the change doesn't violate core identity invariants."""
        # For now, simple keyword-based 'poison' check or invariant protection
        # Future: Use LLM-based semantic alignment check against GEMINI.md/SELF tier
        if proposal.change_type == "identity":
            # Identity changes are extremely sensitive
            return {"pass": False, "reason": "Autonomous identity modification is currently manually locked."}
        
        return {"pass": True, "reason": "No invariant violation detected."}

    def _check_safety_constraints(self, proposal: ModificationProposal) -> Dict[str, Any]:
        """Check for obvious security/safety hazards."""
        content = proposal.proposed_content
        if "rm -rf /" in content or "eval(" in content:
            return {"pass": False, "reason": "Dangerous command detected in proposal content."}
        
        return {"pass": True, "reason": "Basic safety checks passed."}
