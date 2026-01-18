from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar
from pytorch_lightning.utilities.exceptions import MisconfigurationException
@dataclass
class OutputResult:

    def asdict(self) -> Dict[str, Any]:
        raise NotImplementedError