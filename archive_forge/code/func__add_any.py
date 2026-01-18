import json
import os
from typing import Any, Dict, List, Optional, Union
import wandb
import wandb.data_types as data_types
from wandb.data_types import _SavedModel
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
def _add_any(artifact: Artifact, path_or_obj: Union[str, ArtifactManifestEntry, data_types.WBValue], name: Optional[str]) -> Any:
    """Add an object to an artifact.

    High-level wrapper to add object(s) to an artifact - calls any of the .add* methods
    under Artifact depending on the type of object that's passed in. This will probably
    be moved to the Artifact class in the future.

    Args:
        artifact: `Artifact` - artifact created with `wandb.Artifact(...)`
        path_or_obj: `Union[str, ArtifactManifestEntry, data_types.WBValue]` - either a
            str or valid object which indicates what to add to an artifact.

        name: `str` - the name of the object which is added to an artifact.

    Returns:
        Type[Any] - Union[None, ArtifactManifestEntry, etc]

    """
    if isinstance(path_or_obj, ArtifactManifestEntry):
        return artifact.add_reference(path_or_obj, name)
    elif isinstance(path_or_obj, data_types.WBValue):
        return artifact.add(path_or_obj, name)
    elif isinstance(path_or_obj, str):
        if os.path.isdir(path_or_obj):
            return artifact.add_dir(path_or_obj)
        elif os.path.isfile(path_or_obj):
            return artifact.add_file(path_or_obj)
        else:
            with artifact.new_file(name) as f:
                f.write(json.dumps(path_or_obj, sort_keys=True))
    else:
        raise ValueError(f'Expected `path_or_obj` to be instance of `ArtifactManifestEntry`, `WBValue`, or `str, found {type(path_or_obj)}')