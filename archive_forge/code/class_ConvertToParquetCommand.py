import time
from argparse import ArgumentParser
from typing import Optional
from huggingface_hub import HfApi, create_branch, get_repo_discussions
from datasets import get_dataset_config_names, get_dataset_default_config_name, load_dataset
from datasets.commands import BaseDatasetsCLICommand
class ConvertToParquetCommand(BaseDatasetsCLICommand):

    @staticmethod
    def register_subcommand(parser):
        parser: ArgumentParser = parser.add_parser('convert_to_parquet', help='Convert dataset to Parquet')
        parser.add_argument('dataset_id', help='source dataset ID')
        parser.add_argument('--token', help='access token to the Hugging Face Hub')
        parser.add_argument('--revision', help='source revision')
        parser.add_argument('--trust_remote_code', action='store_true', help='whether to trust the code execution of the load script')
        parser.set_defaults(func=_command_factory)

    def __init__(self, dataset_id: str, token: Optional[str], revision: Optional[str], trust_remote_code: bool):
        self._dataset_id = dataset_id
        self._token = token
        self._revision = revision
        self._trust_remote_code = trust_remote_code

    def run(self) -> None:
        dataset_id = self._dataset_id
        token = self._token
        revision = self._revision
        trust_remote_code = self._trust_remote_code
        print(f'{dataset_id}')
        configs = get_dataset_config_names(dataset_id, token=token, revision=revision, trust_remote_code=trust_remote_code)
        print(f'configs = {configs!r}')
        default_config = get_dataset_default_config_name(dataset_id, token=token, revision=revision, trust_remote_code=trust_remote_code)
        print(f'default_config = {default_config!r}')
        if default_config:
            config = default_config
            configs.remove(default_config)
        else:
            config = configs.pop(0)
        print(f'config = {config!r}')
        dataset = load_dataset(dataset_id, config, revision=revision, trust_remote_code=trust_remote_code)
        commit_info = dataset.push_to_hub(dataset_id, config_name=config, commit_message='Convert dataset to Parquet', commit_description='Convert dataset to Parquet.', create_pr=True, token=token, set_default=default_config is not None)
        time.sleep(5)
        if commit_info:
            pr_revision, pr_url = (commit_info.pr_revision, commit_info.pr_url)
        else:
            pr_revision, pr_url = infer_pr(dataset_id, token=token)
        for config in configs:
            print(f'config = {config!r}')
            dataset = load_dataset(dataset_id, config, revision=revision, trust_remote_code=trust_remote_code)
            dataset.push_to_hub(dataset_id, config_name=config, commit_message=f'Add {config} data files', revision=pr_revision, token=token)
            time.sleep(5)
        delete_files(dataset_id, revision=pr_revision, token=token)
        if not revision:
            create_branch(dataset_id, branch='script', repo_type='dataset', token=token, exist_ok=True)
        print(f'You can find your PR to convert the dataset to Parquet at: {pr_url}')