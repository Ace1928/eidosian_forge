import concurrent.futures
import logging
import os
import queue
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
import wandb.util
from wandb.filesync import stats, step_checksum, step_upload
from wandb.sdk.lib.paths import LogicalPath
Tell the file pusher that a file's changed and should be uploaded.

        Arguments:
            save_name: string logical location of the file relative to the run
                directory.
            path: actual string path of the file to upload on the filesystem.
        