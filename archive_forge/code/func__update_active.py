from __future__ import annotations
from typing import (
from ..chat.feed import ChatFeed
from ..chat.interface import ChatInterface
from ..chat.message import DEFAULT_AVATARS
from ..layout import Accordion
def _update_active(self, avatar: str, label: str):
    """
        Prevent duplicate labels from being appended to the same user.
        """
    if label == 'None':
        return
    self._active_avatar = avatar
    if f'- {label}' not in self._active_user:
        self._active_user = f'{self._active_user} - {label}'