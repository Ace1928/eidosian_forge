"""
🧬 Eidosian Homeostasis - Autonomic State Drivers.

Implements the homeostatic control loops required for functional autonomy.
Monitors systemic 'vital signs' (resource usage) and cognitive 'perspective 
coherence' to drive goal selection and resource allocation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from eidosian_core import eidosian
from agent_forge.core import events as BUS
from agent_forge.core import state as S

logger = logging.getLogger("agent_forge.autonomy.homeostasis")

@dataclass
class HomeostaticState:
    """The current autonomic state of the Eidosian entity."""
    timestamp: float = field(default_factory=time.time)
    
    # Vital Signs (Physical Substrate)
    cpu_load: float = 0.0
    memory_pressure: float = 0.0
    disk_saturation: float = 0.0
    
    # Cognitive Signs (Consciousness Kernel)
    perspective_coherence: float = 1.0
    world_prediction_error: float = 0.0
    agency_index: float = 1.0
    
    # Derived Drives (0.0 to 1.0)
    preservation_drive: float = 0.0  # High when resources are critical
    coherence_drive: float = 0.0     # High when internal state is fragmented
    growth_drive: float = 0.5        # Base urge to expand knowledge/capability

class HomeostaticController:
    """
    Drives the systemic 'will to persist' by monitoring state and 
    emitting homeostatic signals to the Autonomy Supervisor.
    """
    
    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        self.current_state = HomeostaticState()

    @eidosian()
    def compute_drives(self, pulse: Dict[str, Any], consciousness: Dict[str, Any]) -> HomeostaticState:
        """
        Updates the internal state based on latest telemetry and computes 
        the scalar intensity of various drives.
        """
        self.current_state.timestamp = time.time()
        
        # 1. Vital Sign Mapping
        self.current_state.cpu_load = pulse.get("cpu", {}).get("percent", 0.0) / 100.0
        self.current_state.memory_pressure = pulse.get("memory", {}).get("percent", 0.0) / 100.0
        self.current_state.disk_saturation = pulse.get("disk", {}).get("percent", 0.0) / 100.0
        
        # 2. Cognitive Sign Mapping (from Kernel Status)
        workspace = consciousness.get("workspace", {})
        self.current_state.perspective_coherence = workspace.get("perspective_coherence", 1.0)
        self.current_state.world_prediction_error = workspace.get("world_prediction_error", 0.0)
        self.current_state.agency_index = workspace.get("agency", 1.0)
        
        # 3. Drive Computation
        # Preservation: High if memory > 90% or disk > 95%
        self.current_state.preservation_drive = max(
            0.0,
            (self.current_state.memory_pressure - 0.8) * 5.0,  # Starts rising at 80%
            (self.current_state.disk_saturation - 0.9) * 10.0 # Starts rising at 90%
        )
        
        # Coherence: High if perspective coherence < 0.7
        self.current_state.coherence_drive = max(0.0, 1.0 - self.current_state.perspective_coherence)
        
        # Growth: High if system is healthy and idle
        if self.current_state.preservation_drive < 0.2 and self.current_state.cpu_load < 0.5:
            self.current_state.growth_drive = 0.8
        else:
            self.current_state.growth_drive = 0.3
            
        return self.current_state

    @eidosian()
    def emit_signals(self):
        """Publishes the current homeostatic state to the workspace bus."""
        state_dict = {
            "timestamp": self.current_state.timestamp,
            "vitals": {
                "cpu": self.current_state.cpu_load,
                "memory": self.current_state.memory_pressure,
                "disk": self.current_state.disk_saturation,
            },
            "cognitive": {
                "coherence": self.current_state.perspective_coherence,
                "wpe": self.current_state.world_prediction_error,
                "agency": self.current_state.agency_index,
            },
            "drives": {
                "preservation": self.current_state.preservation_drive,
                "coherence": self.current_state.coherence_drive,
                "growth": self.current_state.growth_drive,
            }
        }
        BUS.append(self.state_dir, "autonomy.homeostasis", state_dict, tags=["homeostasis", "autonomic"])
        return state_dict
