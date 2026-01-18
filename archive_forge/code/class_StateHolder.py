from __future__ import annotations
import datetime
import os
import threading
from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterator
class StateHolder:

    def __init__(self):
        self.capacity = 10000
        self.session_data: OrderedDict[str, SessionState] = OrderedDict()
        self.time_last_used: dict[str, datetime.datetime] = {}
        self.lock = threading.Lock()

    def set_blocks(self, blocks: Blocks):
        self.blocks = blocks
        self.capacity = blocks.state_session_capacity

    def reset(self, blocks: Blocks):
        """Reset the state holder with new blocks. Used during reload mode."""
        self.session_data = OrderedDict()
        self.set_blocks(blocks)

    def __getitem__(self, session_id: str) -> SessionState:
        if session_id not in self.session_data:
            self.session_data[session_id] = SessionState(self.blocks)
        self.update(session_id)
        self.time_last_used[session_id] = datetime.datetime.now()
        return self.session_data[session_id]

    def __contains__(self, session_id: str):
        return session_id in self.session_data

    def update(self, session_id: str):
        with self.lock:
            if session_id in self.session_data:
                self.session_data.move_to_end(session_id)
            if len(self.session_data) > self.capacity:
                self.session_data.popitem(last=False)

    def delete_all_expired_state(self):
        for session_id in self.session_data:
            self.delete_state(session_id, expired_only=True)

    def delete_state(self, session_id: str, expired_only: bool=False):
        if session_id not in self.session_data:
            return
        to_delete = []
        session_state = self.session_data[session_id]
        for component, value, expired in session_state.state_components:
            if not expired_only or expired:
                component.delete_callback(value)
                to_delete.append(component._id)
        for component in to_delete:
            del session_state._data[component]