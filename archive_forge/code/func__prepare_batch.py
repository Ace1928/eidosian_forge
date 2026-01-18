import asyncio
import functools
import queue
import threading
import time
from typing import (
def _prepare_batch(self, batch: Sequence[RequestPrepare]) -> Mapping[str, 'CreateArtifactFilesResponseFile']:
    """Execute the prepareFiles API call.

        Arguments:
            batch: List of RequestPrepare objects
        Returns:
            dict of (save_name: ResponseFile) pairs where ResponseFile is a dict with
                an uploadUrl key. The value of the uploadUrl key is None if the file
                already exists, or a url string if the file should be uploaded.
        """
    return self._api.create_artifact_files([req.file_spec for req in batch])