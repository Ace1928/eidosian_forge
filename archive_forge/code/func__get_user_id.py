import logging
from typing import Optional
import wandb
from wandb.util import (
from . import wandb_helper
from .lib import config_util
def _get_user_id(self, user) -> int:
    if user not in self._users:
        self._users[user] = self._users_cnt
        self._users_inv[self._users_cnt] = user
        object.__setattr__(self, '_users_cnt', self._users_cnt + 1)
    return self._users[user]