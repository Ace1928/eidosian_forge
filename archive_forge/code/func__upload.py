import os
import time
import warnings
from argparse import Namespace, _SubParsersAction
from typing import List, Optional
from huggingface_hub import logging
from huggingface_hub._commit_scheduler import CommitScheduler
from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
from huggingface_hub.hf_api import HfApi
from huggingface_hub.utils import RevisionNotFoundError, disable_progress_bars, enable_progress_bars
def _upload(self) -> str:
    if os.path.isfile(self.local_path):
        if self.include is not None and len(self.include) > 0:
            warnings.warn('Ignoring `--include` since a single file is uploaded.')
        if self.exclude is not None and len(self.exclude) > 0:
            warnings.warn('Ignoring `--exclude` since a single file is uploaded.')
        if self.delete is not None and len(self.delete) > 0:
            warnings.warn('Ignoring `--delete` since a single file is uploaded.')
    if not HF_HUB_ENABLE_HF_TRANSFER:
        logger.info('Consider using `hf_transfer` for faster uploads. This solution comes with some limitations. See https://huggingface.co/docs/huggingface_hub/hf_transfer for more details.')
    if self.every is not None:
        if os.path.isfile(self.local_path):
            folder_path = os.path.dirname(self.local_path)
            path_in_repo = self.path_in_repo[:-len(self.local_path)] if self.path_in_repo.endswith(self.local_path) else self.path_in_repo
            allow_patterns = [self.local_path]
            ignore_patterns = []
        else:
            folder_path = self.local_path
            path_in_repo = self.path_in_repo
            allow_patterns = self.include or []
            ignore_patterns = self.exclude or []
            if self.delete is not None and len(self.delete) > 0:
                warnings.warn('Ignoring `--delete` when uploading with scheduled commits.')
        scheduler = CommitScheduler(folder_path=folder_path, repo_id=self.repo_id, repo_type=self.repo_type, revision=self.revision, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, path_in_repo=path_in_repo, private=self.private, every=self.every, hf_api=self.api)
        print(f'Scheduling commits every {self.every} minutes to {scheduler.repo_id}.')
        try:
            while True:
                time.sleep(100)
        except KeyboardInterrupt:
            scheduler.stop()
            return 'Stopped scheduled commits.'
    if not os.path.isfile(self.local_path) and (not os.path.isdir(self.local_path)):
        raise FileNotFoundError(f"No such file or directory: '{self.local_path}'.")
    repo_id = self.api.create_repo(repo_id=self.repo_id, repo_type=self.repo_type, exist_ok=True, private=self.private, space_sdk='gradio' if self.repo_type == 'space' else None).repo_id
    if self.revision is not None and (not self.create_pr):
        try:
            self.api.repo_info(repo_id=repo_id, repo_type=self.repo_type, revision=self.revision)
        except RevisionNotFoundError:
            logger.info(f"Branch '{self.revision}' not found. Creating it...")
            self.api.create_branch(repo_id=repo_id, repo_type=self.repo_type, branch=self.revision, exist_ok=True)
    if os.path.isfile(self.local_path):
        return self.api.upload_file(path_or_fileobj=self.local_path, path_in_repo=self.path_in_repo, repo_id=repo_id, repo_type=self.repo_type, revision=self.revision, commit_message=self.commit_message, commit_description=self.commit_description, create_pr=self.create_pr)
    else:
        return self.api.upload_folder(folder_path=self.local_path, path_in_repo=self.path_in_repo, repo_id=repo_id, repo_type=self.repo_type, revision=self.revision, commit_message=self.commit_message, commit_description=self.commit_description, create_pr=self.create_pr, allow_patterns=self.include, ignore_patterns=self.exclude, delete_patterns=self.delete)