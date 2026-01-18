import json
import os
from typing import Any, Dict, List, Optional, Union
import wandb
import wandb.data_types as data_types
from wandb.data_types import _SavedModel
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
def _log_artifact_version(name: str, type: str, entries: Dict[str, Union[str, ArtifactManifestEntry, data_types.WBValue]], aliases: Optional[Union[str, List[str]]]=None, description: Optional[str]=None, metadata: Optional[dict]=None, project: Optional[str]=None, scope_project: Optional[bool]=None, job_type: str='auto') -> Artifact:
    """Create an artifact, populate it, and log it with a run.

    If a run is not present, we create one.

    Args:
        name: `str` - name of the artifact. If not scoped to a project, name will be
            suffixed by "-{run_id}".
        type: `str` - type of the artifact, used in the UI to group artifacts of the
            same type.
        entries: `Dict` - dictionary containing the named objects we want added to this
            artifact.
        description: `str` - text description of artifact.
        metadata: `Dict` - users can pass in artifact-specific metadata here, will be
            visible in the UI.
        project: `str` - project under which to place this artifact.
        scope_project: `bool` - if True, we will not suffix `name` with "-{run_id}".
        job_type: `str` - Only applied if run is not present and we create one.
            Used to identify runs of a certain job type, i.e "evaluation".

    Returns:
        Artifact

    """
    if wandb.run is None:
        run = wandb.init(project=project, job_type=job_type, settings=wandb.Settings(silent='true'))
    else:
        run = wandb.run
    if not scope_project:
        name = f'{name}-{run.id}'
    if metadata is None:
        metadata = {}
    art = wandb.Artifact(name, type, description, metadata, False, None)
    for path in entries:
        _add_any(art, entries[path], path)
    aliases = wandb.util._resolve_aliases(aliases)
    run.log_artifact(art, aliases=aliases)
    return art