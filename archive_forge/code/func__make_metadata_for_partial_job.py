import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import wandb
from wandb.apis.internal import Api
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.launch.builder.build import get_current_python_version
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.utils import _is_git_uri
from wandb.sdk.lib import filesystem
from wandb.util import make_artifact_name_safe
def _make_metadata_for_partial_job(job_type: str, tempdir: tempfile.TemporaryDirectory, git_hash: Optional[str], runtime: Optional[str], path: str, entrypoint: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    """Create metadata for partial jobs, return metadata and requirements."""
    metadata = {'_partial': 'v0'}
    if job_type == 'git':
        repo_metadata = _create_repo_metadata(path=path, tempdir=tempdir.name, entrypoint=entrypoint, git_hash=git_hash, runtime=runtime)
        if not repo_metadata:
            tempdir.cleanup()
            return (None, None)
        metadata.update(repo_metadata)
        return (metadata, None)
    if job_type == 'code':
        path, entrypoint = _handle_artifact_entrypoint(path, entrypoint)
        if not entrypoint:
            wandb.termerror('Artifact jobs must have an entrypoint, either included in the path or specified with -E')
            return (None, None)
        artifact_metadata, requirements = _create_artifact_metadata(path=path, entrypoint=entrypoint, runtime=runtime)
        if not artifact_metadata:
            return (None, None)
        metadata.update(artifact_metadata)
        return (metadata, requirements)
    if job_type == 'image':
        if runtime:
            wandb.termwarn('Setting runtime is not supported for image jobs, ignoring runtime')
        if entrypoint:
            wandb.termwarn('Setting an entrypoint is not currently supported for image jobs, ignoring entrypoint argument')
        metadata.update({'python': runtime or '', 'docker': path})
        return (metadata, None)
    wandb.termerror(f'Invalid job type: {job_type}')
    return (None, None)