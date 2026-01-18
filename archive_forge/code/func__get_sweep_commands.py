import logging
from pprint import pformat as pf
from typing import Any, Dict, List, Optional
import wandb
from wandb.sdk.launch.sweeps.scheduler import LOG_PREFIX, RunState, Scheduler, SweepRun
def _get_sweep_commands(self, worker_id: int) -> List[Dict[str, Any]]:
    """Helper to recieve sweep command from backend."""
    _run_states: Dict[str, bool] = {}
    for run_id, run in self._yield_runs():
        if run.worker_id == worker_id and run.state.is_alive:
            _run_states[run_id] = True
    _logger.debug(f'Sending states: \n{pf(_run_states)}\n')
    commands: List[Dict[str, Any]] = self._api.agent_heartbeat(agent_id=self._workers[worker_id].agent_id, metrics={}, run_states=_run_states)
    _logger.debug(f'AgentHeartbeat commands: \n{pf(commands)}\n')
    return commands