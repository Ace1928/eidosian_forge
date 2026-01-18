from __future__ import annotations
import time
import typing as t
from traitlets import Bool, Dict, Float, Instance, Integer, default
from traitlets.config.configurable import LoggingConfigurable
def _callbacks_default(self) -> dict[str, list]:
    return {'restart': [], 'dead': []}