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
@document()
class HuggingFaceDatasetSaver(FlaggingCallback):
    """
    A callback that saves each flagged sample (both the input and output data) to a HuggingFace dataset.

    Example:
        import gradio as gr
        hf_writer = gr.HuggingFaceDatasetSaver(HF_API_TOKEN, "image-classification-mistakes")
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            allow_flagging="manual", flagging_callback=hf_writer)
    Guides: using-flagging
    """

    def __init__(self, hf_token: str, dataset_name: str, private: bool=False, info_filename: str='dataset_info.json', separate_dirs: bool=False):
        """
        Parameters:
            hf_token: The HuggingFace token to use to create (and write the flagged sample to) the HuggingFace dataset (defaults to the registered one).
            dataset_name: The repo_id of the dataset to save the data to, e.g. "image-classifier-1" or "username/image-classifier-1".
            private: Whether the dataset should be private (defaults to False).
            info_filename: The name of the file to save the dataset info (defaults to "dataset_infos.json").
            separate_dirs: If True, each flagged item will be saved in a separate directory. This makes the flagging more robust to concurrent editing, but may be less convenient to use.
        """
        self.hf_token = hf_token
        self.dataset_id = dataset_name
        self.dataset_private = private
        self.info_filename = info_filename
        self.separate_dirs = separate_dirs

    def setup(self, components: list[Component], flagging_dir: str):
        """
        Params:
        flagging_dir (str): local directory where the dataset is cloned,
        updated, and pushed from.
        """
        self.dataset_id = huggingface_hub.create_repo(repo_id=self.dataset_id, token=self.hf_token, private=self.dataset_private, repo_type='dataset', exist_ok=True).repo_id
        path_glob = '**/*.jsonl' if self.separate_dirs else 'data.csv'
        huggingface_hub.metadata_update(repo_id=self.dataset_id, repo_type='dataset', metadata={'configs': [{'config_name': 'default', 'data_files': [{'split': 'train', 'path': path_glob}]}]}, overwrite=True, token=self.hf_token)
        self.components = components
        self.dataset_dir = Path(flagging_dir).absolute() / self.dataset_id.split('/')[-1]
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.infos_file = self.dataset_dir / self.info_filename
        remote_files = [self.info_filename]
        if not self.separate_dirs:
            remote_files.append('data.csv')
        for filename in remote_files:
            try:
                huggingface_hub.hf_hub_download(repo_id=self.dataset_id, repo_type='dataset', filename=filename, local_dir=self.dataset_dir, token=self.hf_token)
            except huggingface_hub.utils.EntryNotFoundError:
                pass

    def flag(self, flag_data: list[Any], flag_option: str='', username: str | None=None) -> int:
        if self.separate_dirs:
            unique_id = str(uuid.uuid4())
            components_dir = self.dataset_dir / unique_id
            data_file = components_dir / 'metadata.jsonl'
            path_in_repo = unique_id
        else:
            components_dir = self.dataset_dir
            data_file = components_dir / 'data.csv'
            path_in_repo = None
        return self._flag_in_dir(data_file=data_file, components_dir=components_dir, path_in_repo=path_in_repo, flag_data=flag_data, flag_option=flag_option, username=username or '')

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

    @staticmethod
    def _save_as_csv(data_file: Path, headers: list[str], row: list[Any]) -> int:
        """Save data as CSV and return the sample name (row number)."""
        is_new = not data_file.exists()
        with data_file.open('a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if is_new:
                writer.writerow(utils.sanitize_list_for_csv(headers))
            writer.writerow(utils.sanitize_list_for_csv(row))
        with data_file.open(encoding='utf-8') as csvfile:
            return sum((1 for _ in csv.reader(csvfile))) - 1

    @staticmethod
    def _save_as_jsonl(data_file: Path, headers: list[str], row: list[Any]) -> str:
        """Save data as JSONL and return the sample name (uuid)."""
        Path.mkdir(data_file.parent, parents=True, exist_ok=True)
        with open(data_file, 'w') as f:
            json.dump(dict(zip(headers, row)), f)
        return data_file.parent.name

    def _deserialize_components(self, data_dir: Path, flag_data: list[Any], flag_option: str='', username: str='') -> tuple[dict[Any, Any], list[Any]]:
        """Deserialize components and return the corresponding row for the flagged sample.

        Images/audio are saved to disk as individual files.
        """
        file_preview_types = {gr.Audio: 'Audio', gr.Image: 'Image'}
        features = OrderedDict()
        row = []
        for component, sample in zip(self.components, flag_data):
            label = component.label or ''
            save_dir = data_dir / client_utils.strip_invalid_filename_characters(label)
            save_dir.mkdir(exist_ok=True, parents=True)
            deserialized = utils.simplify_file_data_in_str(component.flag(sample, save_dir))
            features[label] = {'dtype': 'string', '_type': 'Value'}
            try:
                deserialized_path = Path(deserialized)
                if not deserialized_path.exists():
                    raise FileNotFoundError(f'File {deserialized} not found')
                row.append(str(deserialized_path.relative_to(self.dataset_dir)))
            except (FileNotFoundError, TypeError, ValueError):
                deserialized = '' if deserialized is None else str(deserialized)
                row.append(deserialized)
            if isinstance(component, tuple(file_preview_types)):
                for _component, _type in file_preview_types.items():
                    if isinstance(component, _component):
                        features[label + ' file'] = {'_type': _type}
                        break
                if deserialized:
                    path_in_repo = str(Path(deserialized).relative_to(self.dataset_dir)).replace('\\', '/')
                    row.append(huggingface_hub.hf_hub_url(repo_id=self.dataset_id, filename=path_in_repo, repo_type='dataset'))
                else:
                    row.append('')
        features['flag'] = {'dtype': 'string', '_type': 'Value'}
        features['username'] = {'dtype': 'string', '_type': 'Value'}
        row.append(flag_option)
        row.append(username)
        return (features, row)