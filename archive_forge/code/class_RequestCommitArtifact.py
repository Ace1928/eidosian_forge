import concurrent.futures
import functools
import os
import queue
import shutil
import threading
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, cast
from wandb.filesync import step_upload
from wandb.sdk.lib import filesystem, runid
from wandb.sdk.lib.paths import LogicalPath
class RequestCommitArtifact(NamedTuple):
    artifact_id: str
    finalize: bool
    before_commit: step_upload.PreCommitFn
    result_future: 'concurrent.futures.Future[None]'