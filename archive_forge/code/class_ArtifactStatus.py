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
class ArtifactStatus(TypedDict):
    finalize: bool
    pending_count: int
    commit_requested: bool
    pre_commit_callbacks: MutableSet['PreCommitFn']
    result_futures: MutableSet['concurrent.futures.Future[None]']