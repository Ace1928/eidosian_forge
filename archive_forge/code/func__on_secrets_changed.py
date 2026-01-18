from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from streamlit.runtime.secrets import AttrDict, secrets_singleton
from streamlit.util import calc_md5
def _on_secrets_changed(self, _) -> None:
    """Reset the raw connection object when this connection's secrets change.

        We don't expect either user scripts or connection authors to have to use or
        overwrite this method.
        """
    new_hash = calc_md5(json.dumps(self._secrets.to_dict()))
    if new_hash != self._config_section_hash:
        self._config_section_hash = new_hash
        self.reset()