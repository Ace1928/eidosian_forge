from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from wandb import util
def _set_artifact_target(self, artifact: 'Artifact', name: Optional[str]=None) -> None:
    assert self._artifact_target is None, 'Cannot update artifact_target. Existing target: {}/{}'.format(self._artifact_target.artifact, self._artifact_target.name)
    self._artifact_target = _WBValueArtifactTarget(artifact, name)