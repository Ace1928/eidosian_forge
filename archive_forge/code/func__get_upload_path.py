import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
def _get_upload_path(self) -> str:
    """Formats _upload_path with object attributes.

        Returns:
            The upload path
        """
    if TYPE_CHECKING:
        assert isinstance(self._upload_path, str)
    data = self.attributes
    return self._upload_path.format(**data)