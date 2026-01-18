import json
from typing import Dict, List, Optional
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils.env import get_from_dict_or_env
from langchain_community.tools.connery.models import Action
from langchain_community.tools.connery.tool import ConneryAction
def _get_action(self, action_id: str) -> Action:
    """
        Returns the specified action available in the Connery Runner.
        Parameters:
            action_id (str): The ID of the action to return.
        Returns:
            Action: The action with the specified ID.
        """
    actions = self._list_actions()
    action = next((action for action in actions if action.id == action_id), None)
    if not action:
        raise ValueError(f'The action with ID {action_id} was not found in the listof available actions in the Connery Runner.')
    return action