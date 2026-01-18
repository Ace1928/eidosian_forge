from typing import TYPE_CHECKING, Optional
from wandb import errors
class ArtifactFinalizedError(ArtifactStatusError):
    """Raised for Artifact methods or attributes that can't be changed after logging."""

    def __init__(self, artifact: Optional['Artifact']=None, attr: Optional[str]=None):
        super().__init__(artifact, attr, "'{method_id}' used on logged artifact. Can't modify finalized artifact.")