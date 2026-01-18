from __future__ import annotations
import csv
import datetime
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any
import filelock
import huggingface_hub
from gradio_client import utils as client_utils
from gradio_client.documentation import document
import gradio as gr
from gradio import utils
def _flag_in_dir(self, data_file: Path, components_dir: Path, path_in_repo: str | None, flag_data: list[Any], flag_option: str='', username: str='') -> int:
    features, row = self._deserialize_components(components_dir, flag_data, flag_option, username)
    with filelock.FileLock(str(self.infos_file) + '.lock'):
        if not self.infos_file.exists():
            self.infos_file.write_text(json.dumps({'flagged': {'features': features}}))
            huggingface_hub.upload_file(repo_id=self.dataset_id, repo_type='dataset', token=self.hf_token, path_in_repo=self.infos_file.name, path_or_fileobj=self.infos_file)
    headers = list(features.keys())
    if not self.separate_dirs:
        with filelock.FileLock(components_dir / '.lock'):
            sample_nb = self._save_as_csv(data_file, headers=headers, row=row)
            sample_name = str(sample_nb)
            huggingface_hub.upload_folder(repo_id=self.dataset_id, repo_type='dataset', commit_message=f'Flagged sample #{sample_name}', path_in_repo=path_in_repo, ignore_patterns='*.lock', folder_path=components_dir, token=self.hf_token)
    else:
        sample_name = self._save_as_jsonl(data_file, headers=headers, row=row)
        sample_nb = len([path for path in self.dataset_dir.iterdir() if path.is_dir()])
        huggingface_hub.upload_folder(repo_id=self.dataset_id, repo_type='dataset', commit_message=f'Flagged sample #{sample_name}', path_in_repo=path_in_repo, ignore_patterns='*.lock', folder_path=components_dir, token=self.hf_token)
    return sample_nb