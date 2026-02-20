from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Also add the agent_forge subdir so tests can import `from core` directly
AGENT_FORGE_ROOT = SRC_ROOT / "agent_forge"
if str(AGENT_FORGE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_FORGE_ROOT))
