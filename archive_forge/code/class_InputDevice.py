import uuid
from typing import Optional
class InputDevice:
    """Describes the input device being used for the action."""

    def __init__(self, name: Optional[str]=None):
        self.name = name or uuid.uuid4()
        self.actions = []

    def add_action(self, action):
        """"""
        self.actions.append(action)

    def clear_actions(self):
        self.actions = []

    def create_pause(self, duration: int=0):
        pass