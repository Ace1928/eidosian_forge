from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum
@dataclass
class SafetyCategory:
    name: str
    description: str