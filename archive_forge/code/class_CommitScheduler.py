import atexit
import logging
import os
import time
from concurrent.futures import Future
from dataclasses import dataclass
from io import SEEK_END, SEEK_SET, BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Union
from .hf_api import IGNORE_GIT_FOLDER_PATTERNS, CommitInfo, CommitOperationAdd, HfApi
from .utils import filter_repo_objects
class CommitScheduler:
    """
    Scheduler to upload a local folder to the Hub at regular intervals (e.g. push to hub every 5 minutes).

    The scheduler is started when instantiated and run indefinitely. At the end of your script, a last commit is
    triggered. Checkout the [upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#scheduled-uploads)
    to learn more about how to use it.

    Args:
        repo_id (`str`):
            The id of the repo to commit to.
        folder_path (`str` or `Path`):
            Path to the local folder to upload regularly.
        every (`int` or `float`, *optional*):
            The number of minutes between each commit. Defaults to 5 minutes.
        path_in_repo (`str`, *optional*):
            Relative path of the directory in the repo, for example: `"checkpoints/"`. Defaults to the root folder
            of the repository.
        repo_type (`str`, *optional*):
            The type of the repo to commit to. Defaults to `model`.
        revision (`str`, *optional*):
            The revision of the repo to commit to. Defaults to `main`.
        private (`bool`, *optional*):
            Whether to make the repo private. Defaults to `False`. This value is ignored if the repo already exist.
        token (`str`, *optional*):
            The token to use to commit to the repo. Defaults to the token saved on the machine.
        allow_patterns (`List[str]` or `str`, *optional*):
            If provided, only files matching at least one pattern are uploaded.
        ignore_patterns (`List[str]` or `str`, *optional*):
            If provided, files matching any of the patterns are not uploaded.
        squash_history (`bool`, *optional*):
            Whether to squash the history of the repo after each commit. Defaults to `False`. Squashing commits is
            useful to avoid degraded performances on the repo when it grows too large.
        hf_api (`HfApi`, *optional*):
            The [`HfApi`] client to use to commit to the Hub. Can be set with custom settings (user agent, token,...).

    Example:
    ```py
    >>> from pathlib import Path
    >>> from huggingface_hub import CommitScheduler

    # Scheduler uploads every 10 minutes
    >>> csv_path = Path("watched_folder/data.csv")
    >>> CommitScheduler(repo_id="test_scheduler", repo_type="dataset", folder_path=csv_path.parent, every=10)

    >>> with csv_path.open("a") as f:
    ...     f.write("first line")

    # Some time later (...)
    >>> with csv_path.open("a") as f:
    ...     f.write("second line")
    ```
    """

    def __init__(self, *, repo_id: str, folder_path: Union[str, Path], every: Union[int, float]=5, path_in_repo: Optional[str]=None, repo_type: Optional[str]=None, revision: Optional[str]=None, private: bool=False, token: Optional[str]=None, allow_patterns: Optional[Union[List[str], str]]=None, ignore_patterns: Optional[Union[List[str], str]]=None, squash_history: bool=False, hf_api: Optional['HfApi']=None) -> None:
        self.api = hf_api or HfApi(token=token)
        self.folder_path = Path(folder_path).expanduser().resolve()
        self.path_in_repo = path_in_repo or ''
        self.allow_patterns = allow_patterns
        if ignore_patterns is None:
            ignore_patterns = []
        elif isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]
        self.ignore_patterns = ignore_patterns + IGNORE_GIT_FOLDER_PATTERNS
        if self.folder_path.is_file():
            raise ValueError(f"'folder_path' must be a directory, not a file: '{self.folder_path}'.")
        self.folder_path.mkdir(parents=True, exist_ok=True)
        repo_url = self.api.create_repo(repo_id=repo_id, private=private, repo_type=repo_type, exist_ok=True)
        self.repo_id = repo_url.repo_id
        self.repo_type = repo_type
        self.revision = revision
        self.token = token
        self.last_uploaded: Dict[Path, float] = {}
        if not every > 0:
            raise ValueError(f"'every' must be a positive integer, not '{every}'.")
        self.lock = Lock()
        self.every = every
        self.squash_history = squash_history
        logger.info(f"Scheduled job to push '{self.folder_path}' to '{self.repo_id}' every {self.every} minutes.")
        self._scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        atexit.register(self._push_to_hub)
        self.__stopped = False

    def stop(self) -> None:
        """Stop the scheduler.

        A stopped scheduler cannot be restarted. Mostly for tests purposes.
        """
        self.__stopped = True

    def _run_scheduler(self) -> None:
        """Dumb thread waiting between each scheduled push to Hub."""
        while True:
            self.last_future = self.trigger()
            time.sleep(self.every * 60)
            if self.__stopped:
                break

    def trigger(self) -> Future:
        """Trigger a `push_to_hub` and return a future.

        This method is automatically called every `every` minutes. You can also call it manually to trigger a commit
        immediately, without waiting for the next scheduled commit.
        """
        return self.api.run_as_future(self._push_to_hub)

    def _push_to_hub(self) -> Optional[CommitInfo]:
        if self.__stopped:
            return None
        logger.info('(Background) scheduled commit triggered.')
        try:
            value = self.push_to_hub()
            if self.squash_history:
                logger.info('(Background) squashing repo history.')
                self.api.super_squash_history(repo_id=self.repo_id, repo_type=self.repo_type, branch=self.revision)
            return value
        except Exception as e:
            logger.error(f'Error while pushing to Hub: {e}')
            raise

    def push_to_hub(self) -> Optional[CommitInfo]:
        """
        Push folder to the Hub and return the commit info.

        <Tip warning={true}>

        This method is not meant to be called directly. It is run in the background by the scheduler, respecting a
        queue mechanism to avoid concurrent commits. Making a direct call to the method might lead to concurrency
        issues.

        </Tip>

        The default behavior of `push_to_hub` is to assume an append-only folder. It lists all files in the folder and
        uploads only changed files. If no changes are found, the method returns without committing anything. If you want
        to change this behavior, you can inherit from [`CommitScheduler`] and override this method. This can be useful
        for example to compress data together in a single file before committing. For more details and examples, check
        out our [integration guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#scheduled-uploads).
        """
        with self.lock:
            logger.debug('Listing files to upload for scheduled commit.')
            relpath_to_abspath = {path.relative_to(self.folder_path).as_posix(): path for path in sorted(self.folder_path.glob('**/*')) if path.is_file()}
            prefix = f'{self.path_in_repo.strip('/')}/' if self.path_in_repo else ''
            files_to_upload: List[_FileToUpload] = []
            for relpath in filter_repo_objects(relpath_to_abspath.keys(), allow_patterns=self.allow_patterns, ignore_patterns=self.ignore_patterns):
                local_path = relpath_to_abspath[relpath]
                stat = local_path.stat()
                if self.last_uploaded.get(local_path) is None or self.last_uploaded[local_path] != stat.st_mtime:
                    files_to_upload.append(_FileToUpload(local_path=local_path, path_in_repo=prefix + relpath, size_limit=stat.st_size, last_modified=stat.st_mtime))
        if len(files_to_upload) == 0:
            logger.debug('Dropping schedule commit: no changed file to upload.')
            return None
        logger.debug('Removing unchanged files since previous scheduled commit.')
        add_operations = [CommitOperationAdd(path_or_fileobj=PartialFileIO(file_to_upload.local_path, size_limit=file_to_upload.size_limit), path_in_repo=file_to_upload.path_in_repo) for file_to_upload in files_to_upload]
        logger.debug('Uploading files for scheduled commit.')
        commit_info = self.api.create_commit(repo_id=self.repo_id, repo_type=self.repo_type, operations=add_operations, commit_message='Scheduled Commit', revision=self.revision)
        for file in files_to_upload:
            self.last_uploaded[file.local_path] = file.last_modified
        return commit_info