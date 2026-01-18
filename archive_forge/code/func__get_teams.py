import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
def _get_teams(self) -> List[str]:
    if self._settings and self._settings._offline:
        return []
    if self._server is None:
        self._load_viewer()
    assert self._server is not None
    teams = self._server._viewer.get('teams')
    if teams:
        teams = [team['node']['name'] for team in teams['edges']]
    return teams or []