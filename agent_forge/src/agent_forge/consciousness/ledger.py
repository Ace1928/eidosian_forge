from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from eidosian_core import eidosian

class ContinuityLedger:
    """
    Records and verifies the structural identity of the Eidosian runtime across sessions.
    This provides an empirical trace to answer "Am I the same Eidos?".
    """
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_file = self.log_dir / "continuity_ledger.jsonl"
        self.current_epoch_id = self._generate_epoch_id()
    
    def _generate_epoch_id(self) -> str:
        # A simple temporal-random id for the current continuous run
        return f"epoch_{int(datetime.now().timestamp())}_{os.urandom(4).hex()}"
        
    @eidosian()
    def record_heartbeat(self, state_snapshot: Dict[str, Any]) -> str:
        """
        Record a point in time validating the current continuity state.
        
        state_snapshot should include:
        - active_model
        - memory_stats
        - core_config_hash
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch_id": self.current_epoch_id,
            "state_hash": self._hash_state(state_snapshot),
            "state": state_snapshot
        }
        
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
        return entry["state_hash"]
        
    def _hash_state(self, state: Dict[str, Any]) -> str:
        # Deterministic JSON hash
        serialized = json.dumps(state, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        
    @eidosian()
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.ledger_file.exists():
            return []
        lines = self.ledger_file.read_text().strip().splitlines()
        return [json.loads(line) for line in lines[-limit:]]
