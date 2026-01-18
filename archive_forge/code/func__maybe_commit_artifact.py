import asyncio
import concurrent.futures
import logging
import queue
import sys
import threading
from typing import (
from wandb.errors.term import termerror
from wandb.filesync import upload_job
from wandb.sdk.lib.paths import LogicalPath
def _maybe_commit_artifact(self, artifact_id: str) -> None:
    artifact_status = self._artifacts[artifact_id]
    if artifact_status['pending_count'] == 0 and artifact_status['commit_requested']:
        try:
            for pre_callback in artifact_status['pre_commit_callbacks']:
                pre_callback()
            if artifact_status['finalize']:
                self._api.commit_artifact(artifact_id)
        except Exception as exc:
            termerror(f"Committing artifact failed. Artifact {artifact_id} won't be finalized.")
            termerror(str(exc))
            self._fail_artifact_futures(artifact_id, exc)
        else:
            self._resolve_artifact_futures(artifact_id)