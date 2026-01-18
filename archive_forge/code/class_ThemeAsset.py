from __future__ import annotations
from dataclasses import dataclass, field
import huggingface_hub
import semantic_version
import semantic_version as semver
@dataclass
class ThemeAsset:
    filename: str
    version: semver.Version = field(init=False)

    def __post_init__(self):
        self.version = semver.Version(self.filename.split('@')[1].replace('.json', ''))