import os
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union
import requests
import yaml
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import upload_file
from huggingface_hub.repocard_data import (
from huggingface_hub.utils import get_session, is_jinja_available, yaml_dump
from .constants import REPOCARD_NAME
from .utils import EntryNotFoundError, SoftTemporaryDirectory, logging, validate_hf_hub_args
class RepoCard:
    card_data_class = CardData
    default_template_path = TEMPLATE_MODELCARD_PATH
    repo_type = 'model'

    def __init__(self, content: str, ignore_metadata_errors: bool=False):
        """Initialize a RepoCard from string content. The content should be a
        Markdown file with a YAML block at the beginning and a Markdown body.

        Args:
            content (`str`): The content of the Markdown file.

        Example:
            ```python
            >>> from huggingface_hub.repocard import RepoCard
            >>> text = '''
            ... ---
            ... language: en
            ... license: mit
            ... ---
            ...
            ... # My repo
            ... '''
            >>> card = RepoCard(text)
            >>> card.data.to_dict()
            {'language': 'en', 'license': 'mit'}
            >>> card.text
            '\\n# My repo\\n'

            ```
        <Tip>
        Raises the following error:

            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              when the content of the repo card metadata is not a dictionary.

        </Tip>
        """
        self.ignore_metadata_errors = ignore_metadata_errors
        self.content = content

    @property
    def content(self):
        """The content of the RepoCard, including the YAML block and the Markdown body."""
        line_break = _detect_line_ending(self._content) or '\n'
        return f'---{line_break}{self.data.to_yaml(line_break=line_break)}{line_break}---{line_break}{self.text}'

    @content.setter
    def content(self, content: str):
        """Set the content of the RepoCard."""
        self._content = content
        match = REGEX_YAML_BLOCK.search(content)
        if match:
            yaml_block = match.group(2)
            self.text = content[match.end():]
            data_dict = yaml.safe_load(yaml_block)
            if data_dict is None:
                data_dict = {}
            if not isinstance(data_dict, dict):
                raise ValueError('repo card metadata block should be a dict')
        else:
            logger.warning('Repo card metadata block was not found. Setting CardData to empty.')
            data_dict = {}
            self.text = content
        self.data = self.card_data_class(**data_dict, ignore_metadata_errors=self.ignore_metadata_errors)

    def __str__(self):
        return self.content

    def save(self, filepath: Union[Path, str]):
        """Save a RepoCard to a file.

        Args:
            filepath (`Union[Path, str]`): Filepath to the markdown file to save.

        Example:
            ```python
            >>> from huggingface_hub.repocard import RepoCard
            >>> card = RepoCard("---\\nlanguage: en\\n---\\n# This is a test repo card")
            >>> card.save("/tmp/test.md")

            ```
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            f.write(str(self))

    @classmethod
    def load(cls, repo_id_or_path: Union[str, Path], repo_type: Optional[str]=None, token: Optional[str]=None, ignore_metadata_errors: bool=False):
        """Initialize a RepoCard from a Hugging Face Hub repo's README.md or a local filepath.

        Args:
            repo_id_or_path (`Union[str, Path]`):
                The repo ID associated with a Hugging Face Hub repo or a local filepath.
            repo_type (`str`, *optional*):
                The type of Hugging Face repo to push to. Defaults to None, which will use use "model". Other options
                are "dataset" and "space". Not used when loading from a local filepath. If this is called from a child
                class, the default value will be the child class's `repo_type`.
            token (`str`, *optional*):
                Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
            ignore_metadata_errors (`str`):
                If True, errors while parsing the metadata section will be ignored. Some information might be lost during
                the process. Use it at your own risk.

        Returns:
            [`huggingface_hub.repocard.RepoCard`]: The RepoCard (or subclass) initialized from the repo's
                README.md file or filepath.

        Example:
            ```python
            >>> from huggingface_hub.repocard import RepoCard
            >>> card = RepoCard.load("nateraw/food")
            >>> assert card.data.tags == ["generated_from_trainer", "image-classification", "pytorch"]

            ```
        """
        if Path(repo_id_or_path).exists():
            card_path = Path(repo_id_or_path)
        elif isinstance(repo_id_or_path, str):
            card_path = Path(hf_hub_download(repo_id_or_path, REPOCARD_NAME, repo_type=repo_type or cls.repo_type, token=token))
        else:
            raise ValueError(f'Cannot load RepoCard: path not found on disk ({repo_id_or_path}).')
        with card_path.open(mode='r', newline='', encoding='utf-8') as f:
            return cls(f.read(), ignore_metadata_errors=ignore_metadata_errors)

    def validate(self, repo_type: Optional[str]=None):
        """Validates card against Hugging Face Hub's card validation logic.
        Using this function requires access to the internet, so it is only called
        internally by [`huggingface_hub.repocard.RepoCard.push_to_hub`].

        Args:
            repo_type (`str`, *optional*, defaults to "model"):
                The type of Hugging Face repo to push to. Options are "model", "dataset", and "space".
                If this function is called from a child class, the default will be the child class's `repo_type`.

        <Tip>
        Raises the following errors:

            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if the card fails validation checks.
            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the request to the Hub API fails for any other reason.

        </Tip>
        """
        repo_type = repo_type or self.repo_type
        body = {'repoType': repo_type, 'content': str(self)}
        headers = {'Accept': 'text/plain'}
        try:
            r = get_session().post('https://huggingface.co/api/validate-yaml', body, headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if r.status_code == 400:
                raise ValueError(r.text)
            else:
                raise exc

    def push_to_hub(self, repo_id: str, token: Optional[str]=None, repo_type: Optional[str]=None, commit_message: Optional[str]=None, commit_description: Optional[str]=None, revision: Optional[str]=None, create_pr: Optional[bool]=None, parent_commit: Optional[str]=None):
        """Push a RepoCard to a Hugging Face Hub repo.

        Args:
            repo_id (`str`):
                The repo ID of the Hugging Face Hub repo to push to. Example: "nateraw/food".
            token (`str`, *optional*):
                Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to
                the stored token.
            repo_type (`str`, *optional*, defaults to "model"):
                The type of Hugging Face repo to push to. Options are "model", "dataset", and "space". If this
                function is called by a child class, it will default to the child class's `repo_type`.
            commit_message (`str`, *optional*):
                The summary / title / first line of the generated commit.
            commit_description (`str`, *optional*)
                The description of the generated commit.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the `"main"` branch.
            create_pr (`bool`, *optional*):
                Whether or not to create a Pull Request with this commit. Defaults to `False`.
            parent_commit (`str`, *optional*):
                The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
                If specified and `create_pr` is `False`, the commit will fail if `revision` does not point to `parent_commit`.
                If specified and `create_pr` is `True`, the pull request will be created from `parent_commit`.
                Specifying `parent_commit` ensures the repo has not changed before committing the changes, and can be
                especially useful if the repo is updated / committed to concurrently.
        Returns:
            `str`: URL of the commit which updated the card metadata.
        """
        repo_type = repo_type or self.repo_type
        self.validate(repo_type=repo_type)
        with SoftTemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / REPOCARD_NAME
            tmp_path.write_text(str(self))
            url = upload_file(path_or_fileobj=str(tmp_path), path_in_repo=REPOCARD_NAME, repo_id=repo_id, token=token, repo_type=repo_type, commit_message=commit_message, commit_description=commit_description, create_pr=create_pr, revision=revision, parent_commit=parent_commit)
        return url

    @classmethod
    def from_template(cls, card_data: CardData, template_path: Optional[str]=None, **template_kwargs):
        """Initialize a RepoCard from a template. By default, it uses the default template.

        Templates are Jinja2 templates that can be customized by passing keyword arguments.

        Args:
            card_data (`huggingface_hub.CardData`):
                A huggingface_hub.CardData instance containing the metadata you want to include in the YAML
                header of the repo card on the Hugging Face Hub.
            template_path (`str`, *optional*):
                A path to a markdown file with optional Jinja template variables that can be filled
                in with `template_kwargs`. Defaults to the default template.

        Returns:
            [`huggingface_hub.repocard.RepoCard`]: A RepoCard instance with the specified card data and content from the
            template.
        """
        if is_jinja_available():
            import jinja2
        else:
            raise ImportError('Using RepoCard.from_template requires Jinja2 to be installed. Please install it with `pip install Jinja2`.')
        kwargs = card_data.to_dict().copy()
        kwargs.update(template_kwargs)
        template = jinja2.Template(Path(template_path or cls.default_template_path).read_text())
        content = template.render(card_data=card_data.to_yaml(), **kwargs)
        return cls(content)