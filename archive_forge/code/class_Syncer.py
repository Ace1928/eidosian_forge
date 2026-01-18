import abc
import logging
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.train.constants import _DEPRECATED_VALUE
from ray.util import log_once
from ray.util.annotations import PublicAPI
from ray.widgets import Template
class Syncer(abc.ABC):
    """Syncer class for synchronizing data between Ray nodes and remote (cloud) storage.

    This class handles data transfer for two cases:

    1. Synchronizing data such as experiment checkpoints from the driver to
       cloud storage.
    2. Synchronizing data such as trial checkpoints from remote trainables to
       cloud storage.

    Synchronizing tasks are usually asynchronous and can be awaited using ``wait()``.
    The base class implements a ``wait_or_retry()`` API that will retry a failed
    sync command.

    The base class also exposes an API to only kick off syncs every ``sync_period``
    seconds.

    Args:
        sync_period: The minimum time in seconds between sync operations, as
            used by ``sync_up/down_if_needed``.
        sync_timeout: The maximum time to wait for a sync process to finish before
            issuing a new sync operation. Ex: should be used by ``wait`` if launching
            asynchronous sync tasks.
    """

    def __init__(self, sync_period: float=DEFAULT_SYNC_PERIOD, sync_timeout: float=DEFAULT_SYNC_TIMEOUT):
        self.sync_period = sync_period
        self.sync_timeout = sync_timeout
        self.last_sync_up_time = float('-inf')
        self.last_sync_down_time = float('-inf')

    @abc.abstractmethod
    def sync_up(self, local_dir: str, remote_dir: str, exclude: Optional[List]=None) -> bool:
        """Synchronize local directory to remote directory.

        This function can spawn an asynchronous process that can be awaited in
        ``wait()``.

        Args:
            local_dir: Local directory to sync from.
            remote_dir: Remote directory to sync up to. This is an URI
                (``protocol://remote/path``).
            exclude: Pattern of files to exclude, e.g.
                ``["*/checkpoint_*]`` to exclude trial checkpoints.

        Returns:
            True if sync process has been spawned, False otherwise.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def sync_down(self, remote_dir: str, local_dir: str, exclude: Optional[List]=None) -> bool:
        """Synchronize remote directory to local directory.

        This function can spawn an asynchronous process that can be awaited in
        ``wait()``.

        Args:
            remote_dir: Remote directory to sync down from. This is an URI
                (``protocol://remote/path``).
            local_dir: Local directory to sync to.
            exclude: Pattern of files to exclude, e.g.
                ``["*/checkpoint_*]`` to exclude trial checkpoints.

        Returns:
            True if sync process has been spawned, False otherwise.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, remote_dir: str) -> bool:
        """Delete directory on remote storage.

        This function can spawn an asynchronous process that can be awaited in
        ``wait()``.

        Args:
            remote_dir: Remote directory to delete. This is an URI
                (``protocol://remote/path``).

        Returns:
            True if sync process has been spawned, False otherwise.

        """
        raise NotImplementedError

    def retry(self):
        """Retry the last sync up, sync down, or delete command.

        You should implement this method if you spawn asynchronous syncing
        processes.
        """
        pass

    def wait(self):
        """Wait for asynchronous sync command to finish.

        You should implement this method if you spawn asynchronous syncing
        processes. This method should timeout after the asynchronous command
        has run for `sync_timeout` seconds and raise a `TimeoutError`.
        """
        pass

    def sync_up_if_needed(self, local_dir: str, remote_dir: str, exclude: Optional[List]=None) -> bool:
        """Syncs up if time since last sync up is greater than sync_period.

        Args:
            local_dir: Local directory to sync from.
            remote_dir: Remote directory to sync up to. This is an URI
                (``protocol://remote/path``).
            exclude: Pattern of files to exclude, e.g.
                ``["*/checkpoint_*]`` to exclude trial checkpoints.
        """
        now = time.time()
        if now - self.last_sync_up_time >= self.sync_period:
            result = self.sync_up(local_dir=local_dir, remote_dir=remote_dir, exclude=exclude)
            self.last_sync_up_time = now
            return result

    def sync_down_if_needed(self, remote_dir: str, local_dir: str, exclude: Optional[List]=None):
        """Syncs down if time since last sync down is greater than sync_period.

        Args:
            remote_dir: Remote directory to sync down from. This is an URI
                (``protocol://remote/path``).
            local_dir: Local directory to sync to.
            exclude: Pattern of files to exclude, e.g.
                ``["*/checkpoint_*]`` to exclude trial checkpoints.
        """
        now = time.time()
        if now - self.last_sync_down_time >= self.sync_period:
            result = self.sync_down(remote_dir=remote_dir, local_dir=local_dir, exclude=exclude)
            self.last_sync_down_time = now
            return result

    def wait_or_retry(self, max_retries: int=2, backoff_s: int=5):
        assert max_retries > 0
        last_error_traceback = None
        for i in range(max_retries + 1):
            try:
                self.wait()
            except Exception as e:
                attempts_remaining = max_retries - i
                if attempts_remaining == 0:
                    last_error_traceback = traceback.format_exc()
                    break
                logger.error(f'The latest sync operation failed with the following error: {repr(e)}\nRetrying {attempts_remaining} more time(s) after sleeping for {backoff_s} seconds...')
                time.sleep(backoff_s)
                self.retry()
                continue
            return
        raise RuntimeError(f'Failed sync even after {max_retries} retries. The latest sync failed with the following error:\n{last_error_traceback}')

    def reset(self):
        self.last_sync_up_time = float('-inf')
        self.last_sync_down_time = float('-inf')

    def close(self):
        pass

    def _repr_html_(self) -> str:
        return