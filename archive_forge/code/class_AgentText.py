import os
import pathlib
import tempfile
import uuid
import numpy as np
from ..utils import is_soundfile_availble, is_torch_available, is_vision_available, logging
class AgentText(AgentType, str):
    """
    Text type returned by the agent. Behaves as a string.
    """

    def to_raw(self):
        return self._value

    def to_string(self):
        return self._value