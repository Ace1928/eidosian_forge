from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from wandb import util
class _WBValueArtifactSource:
    artifact: 'Artifact'
    name: Optional[str]

    def __init__(self, artifact: 'Artifact', name: Optional[str]=None) -> None:
        self.artifact = artifact
        self.name = name