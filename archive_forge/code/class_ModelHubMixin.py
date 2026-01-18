import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type, TypeVar, Union, get_args
from .constants import CONFIG_NAME, PYTORCH_WEIGHTS_NAME, SAFETENSORS_SINGLE_FILE
from .file_download import hf_hub_download
from .hf_api import HfApi
from .utils import (
from .utils._deprecation import _deprecate_arguments
class ModelHubMixin:
    """
    A generic mixin to integrate ANY machine learning framework with the Hub.

    To integrate your framework, your model class must inherit from this class. Custom logic for saving/loading models
    have to be overwritten in  [`_from_pretrained`] and [`_save_pretrained`]. [`PyTorchModelHubMixin`] is a good example
    of mixin integration with the Hub. Check out our [integration guide](../guides/integrations) for more instructions.

    Example:

    ```python
    >>> from dataclasses import dataclass
    >>> from huggingface_hub import ModelHubMixin

    # Define your model configuration (optional)
    >>> @dataclass
    ... class Config:
    ...     foo: int = 512
    ...     bar: str = "cpu"

    # Inherit from ModelHubMixin (and optionally from your framework's model class)
    >>> class MyCustomModel(ModelHubMixin):
    ...     def __init__(self, config: Config):
    ...         # define how to initialize your model
    ...         super().__init__()
    ...         ...
    ...
    ...     def _save_pretrained(self, save_directory: Path) -> None:
    ...         # define how to serialize your model
    ...         ...
    ...
    ...     @classmethod
    ...     def from_pretrained(
    ...         cls: Type[T],
    ...         pretrained_model_name_or_path: Union[str, Path],
    ...         *,
    ...         force_download: bool = False,
    ...         resume_download: bool = False,
    ...         proxies: Optional[Dict] = None,
    ...         token: Optional[Union[str, bool]] = None,
    ...         cache_dir: Optional[Union[str, Path]] = None,
    ...         local_files_only: bool = False,
    ...         revision: Optional[str] = None,
    ...         **model_kwargs,
    ...     ) -> T:
    ...         # define how to deserialize your model
    ...         ...

    >>> model = MyCustomModel(config=Config(foo=256, bar="gpu"))

    # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")

    # Push model weights to the Hub
    >>> model.push_to_hub("my-awesome-model")

    # Download and initialize weights from the Hub
    >>> reloaded_model = MyCustomModel.from_pretrained("username/my-awesome-model")
    >>> reloaded_model.config
    Config(foo=256, bar="gpu")
    ```
    """
    config: Optional[Union[dict, 'DataclassInstance']] = None

    def __new__(cls, *args, **kwargs) -> 'ModelHubMixin':
        instance = super().__new__(cls)
        if instance.config is None:
            if 'config' in kwargs:
                instance.config = kwargs['config']
            elif len(args) > 0:
                sig = inspect.signature(cls.__init__)
                parameters = list(sig.parameters)[1:]
                for key, value in zip(parameters, args):
                    if key == 'config':
                        instance.config = value
                        break
        return instance

    def save_pretrained(self, save_directory: Union[str, Path], *, config: Optional[Union[dict, 'DataclassInstance']]=None, repo_id: Optional[str]=None, push_to_hub: bool=False, **push_to_hub_kwargs) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            kwargs:
                Additional key word arguments passed along to the [`~ModelHubMixin.push_to_hub`] method.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self._save_pretrained(save_directory)
        if config is None:
            config = self.config
        if config is not None:
            if is_dataclass(config):
                config = asdict(config)
            (save_directory / CONFIG_NAME).write_text(json.dumps(config, indent=2))
        if push_to_hub:
            kwargs = push_to_hub_kwargs.copy()
            if config is not None:
                kwargs['config'] = config
            if repo_id is None:
                repo_id = save_directory.name
            return self.push_to_hub(repo_id=repo_id, **kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        Overwrite this method in subclass to define how to save your model.
        Check out our [integration guide](../guides/integrations) for instructions.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls: Type[T], pretrained_model_name_or_path: Union[str, Path], *, force_download: bool=False, resume_download: bool=False, proxies: Optional[Dict]=None, token: Optional[Union[str, bool]]=None, cache_dir: Optional[Union[str, Path]]=None, local_files_only: bool=False, revision: Optional[str]=None, **model_kwargs) -> T:
        """
        Download a model from the Huggingface Hub and instantiate it.

        Args:
            pretrained_model_name_or_path (`str`, `Path`):
                - Either the `model_id` (string) of a model hosted on the Hub, e.g. `bigscience/bloom`.
                - Or a path to a `directory` containing model weights saved using
                    [`~transformers.PreTrainedModel.save_pretrained`], e.g., `../path/to/my_model_directory/`.
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on every request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs (`Dict`, *optional*):
                Additional kwargs to pass to the model during initialization.
        """
        model_id = str(pretrained_model_name_or_path)
        config_file: Optional[str] = None
        if os.path.isdir(model_id):
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.warning(f'{CONFIG_NAME} not found in {Path(model_id).resolve()}')
        else:
            try:
                config_file = hf_hub_download(repo_id=model_id, filename=CONFIG_NAME, revision=revision, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, token=token, local_files_only=local_files_only)
            except HfHubHTTPError as e:
                logger.info(f'{CONFIG_NAME} not found on the HuggingFace Hub: {str(e)}')
        config = None
        if config_file is not None:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            init_parameters = inspect.signature(cls.__init__).parameters
            if 'config' in init_parameters:
                config_annotation = init_parameters['config'].annotation
                if config_annotation is inspect.Parameter.empty:
                    pass
                elif is_dataclass(config_annotation):
                    config = config_annotation(**config)
                else:
                    for _sub_annotation in get_args(config_annotation):
                        if is_dataclass(_sub_annotation):
                            config = _sub_annotation(**config)
                            break
                model_kwargs['config'] = config
            elif any((param.kind == inspect.Parameter.VAR_KEYWORD for param in init_parameters.values())):
                model_kwargs['config'] = config
        instance = cls._from_pretrained(model_id=str(model_id), revision=revision, cache_dir=cache_dir, force_download=force_download, proxies=proxies, resume_download=resume_download, local_files_only=local_files_only, token=token, **model_kwargs)
        if config is not None and instance.config is None:
            instance.config = config
        return instance

    @classmethod
    def _from_pretrained(cls: Type[T], *, model_id: str, revision: Optional[str], cache_dir: Optional[Union[str, Path]], force_download: bool, proxies: Optional[Dict], resume_download: bool, local_files_only: bool, token: Optional[Union[str, bool]], **model_kwargs) -> T:
        """Overwrite this method in subclass to define how to load your model from pretrained.

        Use [`hf_hub_download`] or [`snapshot_download`] to download files from the Hub before loading them. Most
        args taken as input can be directly passed to those 2 methods. If needed, you can add more arguments to this
        method using "model_kwargs". For example [`PyTorchModelHubMixin._from_pretrained`] takes as input a `map_location`
        parameter to set on which device the model should be loaded.

        Check out our [integration guide](../guides/integrations) for more instructions.

        Args:
            model_id (`str`):
                ID of the model to load from the Huggingface Hub (e.g. `bigscience/bloom`).
            revision (`str`, *optional*):
                Revision of the model on the Hub. Can be a branch name, a git tag or any commit id. Defaults to the
                latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the model weights and configuration files from the Hub, overriding
                the existing cache.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether to delete incompletely received files. Will attempt to resume the download if such a file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint (e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            model_kwargs:
                Additional keyword arguments passed along to the [`~ModelHubMixin._from_pretrained`] method.
        """
        raise NotImplementedError

    @_deprecate_arguments(version='0.23.0', deprecated_args=['api_endpoint'], custom_message='Use `HF_ENDPOINT` environment variable instead.')
    @validate_hf_hub_args
    def push_to_hub(self, repo_id: str, *, config: Optional[Union[dict, 'DataclassInstance']]=None, commit_message: str='Push model using huggingface_hub.', private: bool=False, token: Optional[str]=None, branch: Optional[str]=None, create_pr: Optional[bool]=None, allow_patterns: Optional[Union[List[str], str]]=None, ignore_patterns: Optional[Union[List[str], str]]=None, delete_patterns: Optional[Union[List[str], str]]=None, api_endpoint: Optional[str]=None) -> str:
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
        `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
        details.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit. Defaults to `False`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.
            delete_patterns (`List[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo.

        Returns:
            The url of the commit of your model in the given repository.
        """
        api = HfApi(endpoint=api_endpoint, token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id
        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, config=config)
            return api.upload_folder(repo_id=repo_id, repo_type='model', folder_path=saved_path, commit_message=commit_message, revision=branch, create_pr=create_pr, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns, delete_patterns=delete_patterns)