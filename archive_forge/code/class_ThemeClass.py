from __future__ import annotations
import json
import re
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import Iterable
import huggingface_hub
import semantic_version as semver
from gradio_client.documentation import document
from huggingface_hub import CommitOperationAdd
from gradio.themes.utils import (
from gradio.themes.utils.readme_content import README_CONTENT
class ThemeClass:

    def __init__(self):
        self._stylesheets = []
        self.name = None

    def _get_theme_css(self):
        css = {}
        dark_css = {}
        for attr, val in self.__dict__.items():
            if attr.startswith('_'):
                continue
            if val is None:
                if attr.endswith('_dark'):
                    dark_css[attr[:-5]] = None
                    continue
                else:
                    raise ValueError(f"Cannot set '{attr}' to None - only dark mode variables can be None.")
            val = str(val)
            pattern = '(\\*)([\\w_]+)(\\b)'

            def repl_func(match):
                full_match = match.group(0)
                if full_match.startswith('*') and full_match.endswith('_dark'):
                    raise ValueError(f"Cannot refer '{attr}' to '{val}' - dark variable references are automatically used for dark mode attributes, so do not use the _dark suffix in the value.")
                if attr.endswith('_dark') and full_match.startswith('*') and (attr[:-5] == full_match[1:]):
                    raise ValueError(f"Cannot refer '{attr}' to '{val}' - if dark and light mode values are the same, set dark mode version to None.")
                word = match.group(2)
                word = word.replace('_', '-')
                return f'var(--{word})'
            val = re.sub(pattern, repl_func, val)
            attr = attr.replace('_', '-')
            if attr.endswith('-dark'):
                attr = attr[:-5]
                dark_css[attr] = val
            else:
                css[attr] = val
        for attr, val in css.items():
            if attr not in dark_css:
                dark_css[attr] = val
        css_code = ':root {\n' + '\n'.join([f'  --{attr}: {val};' for attr, val in css.items()]) + '\n}'
        dark_css_code = '.dark {\n' + '\n'.join([f'  --{attr}: {val};' for attr, val in dark_css.items()]) + '\n}'
        return f'{css_code}\n{dark_css_code}'

    def _get_computed_value(self, property: str, depth=0) -> str:
        max_depth = 100
        if depth > max_depth:
            warnings.warn(f"Cannot resolve '{property}' - circular reference detected.")
            return ''
        is_dark = property.endswith('_dark')
        if is_dark:
            set_value = getattr(self, property, getattr(self, property[:-5], ''))
        else:
            set_value = getattr(self, property, '')
        pattern = '(\\*)([\\w_]+)(\\b)'

        def repl_func(match, depth):
            word = match.group(2)
            dark_suffix = '_dark' if property.endswith('_dark') else ''
            return self._get_computed_value(word + dark_suffix, depth + 1)
        computed_value = re.sub(pattern, lambda match: repl_func(match, depth), set_value)
        return computed_value

    def to_dict(self):
        """Convert the theme into a python dictionary."""
        schema = {'theme': {}}
        for prop in dir(self):
            if (not prop.startswith('_') or prop.startswith('_font') or prop == '_stylesheets' or (prop == 'name')) and isinstance(getattr(self, prop), (list, str)):
                schema['theme'][prop] = getattr(self, prop)
        return schema

    @classmethod
    def load(cls, path: str) -> ThemeClass:
        """Load a theme from a json file.

        Parameters:
            path: The filepath to read.
        """
        with open(path) as fp:
            return cls.from_dict(json.load(fp, object_hook=fonts.as_font))

    @classmethod
    def from_dict(cls, theme: dict[str, dict[str, str]]) -> ThemeClass:
        """Create a theme instance from a dictionary representation.

        Parameters:
            theme: The dictionary representation of the theme.
        """
        new_theme = cls()
        for prop, value in theme['theme'].items():
            setattr(new_theme, prop, value)
        base = Base()
        for attr in base.__dict__:
            if not attr.startswith('_') and (not hasattr(new_theme, attr)):
                setattr(new_theme, attr, getattr(base, attr))
        return new_theme

    def dump(self, filename: str):
        """Write the theme to a json file.

        Parameters:
            filename: The path to write the theme too
        """
        Path(filename).write_text(json.dumps(self.to_dict(), cls=fonts.FontEncoder))

    @classmethod
    def from_hub(cls, repo_name: str, hf_token: str | None=None):
        """Load a theme from the hub.

        This DOES NOT require a HuggingFace account for downloading publicly available themes.

        Parameters:
            repo_name: string of the form <author>/<theme-name>@<semantic-version-expression>.  If a semantic version expression is omitted, the latest version will be fetched.
            hf_token: HuggingFace Token. Only needed to download private themes.
        """
        if '@' not in repo_name:
            name, version = (repo_name, None)
        else:
            name, version = repo_name.split('@')
        api = huggingface_hub.HfApi(token=hf_token)
        try:
            space_info = api.space_info(name)
        except huggingface_hub.utils._errors.RepositoryNotFoundError as e:
            raise ValueError(f'The space {name} does not exist') from e
        assets = get_theme_assets(space_info)
        matching_version = get_matching_version(assets, version)
        if not matching_version:
            raise ValueError(f'Cannot find a matching version for expression {version} from files {[f.filename for f in assets]}')
        theme_file = huggingface_hub.hf_hub_download(repo_id=name, repo_type='space', filename=f'themes/theme_schema@{matching_version.version}.json')
        theme = cls.load(theme_file)
        theme.name = name
        return theme

    @staticmethod
    def _get_next_version(space_info: huggingface_hub.hf_api.SpaceInfo) -> str:
        assets = get_theme_assets(space_info)
        latest_version = max(assets, key=lambda asset: asset.version).version
        return str(latest_version.next_patch())

    @staticmethod
    def _theme_version_exists(space_info: huggingface_hub.hf_api.SpaceInfo, version: str) -> bool:
        assets = get_theme_assets(space_info)
        return any((a.version == semver.Version(version) for a in assets))

    def push_to_hub(self, repo_name: str, org_name: str | None=None, version: str | None=None, hf_token: str | None=None, theme_name: str | None=None, description: str | None=None, private: bool=False):
        """Upload a theme to the HuggingFace hub.

        This requires a HuggingFace account.

        Parameters:
            repo_name: The name of the repository to store the theme assets, e.g. 'my_theme' or 'sunset'.
            org_name: The name of the org to save the space in. If None (the default), the username corresponding to the logged in user, or hƒ_token is used.
            version: A semantic version tag for theme. Bumping the version tag lets you publish updates to a theme without changing the look of applications that already loaded your theme.
            hf_token: API token for your HuggingFace account
            theme_name: Name for the name. If None, defaults to repo_name
            description: A long form description to your theme.
        """
        from gradio import __version__
        api = huggingface_hub.HfApi()
        if not hf_token:
            try:
                author = huggingface_hub.whoami()['name']
            except OSError as e:
                raise ValueError('In order to push to hub, log in via `huggingface-cli login` or provide a theme_token to push_to_hub. For more information see https://huggingface.co/docs/huggingface_hub/quick-start#login') from e
        else:
            author = huggingface_hub.whoami(token=hf_token)['name']
        space_id = f'{org_name or author}/{repo_name}'
        try:
            space_info = api.space_info(space_id)
        except Exception:
            space_info = None
        space_exists = space_info is not None
        if not version:
            version = self._get_next_version(space_info) if space_exists else '0.0.1'
        else:
            _ = semver.Version(version)
        if space_exists and self._theme_version_exists(space_info, version):
            raise ValueError(f'The space {space_id} already has a theme with version {version}. See: themes/theme_schema@{version}.json. To manually override this version, use the HuggingFace hub UI.')
        theme_name = theme_name or repo_name
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as css_file:
            contents = self.to_dict()
            contents['version'] = version
            json.dump(contents, css_file, cls=fonts.FontEncoder)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as readme_file:
            readme_content = README_CONTENT.format(theme_name=theme_name, description=description or 'Add a description of this theme here!', author=author, gradio_version=__version__)
            readme_file.write(textwrap.dedent(readme_content))
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as app_file:
            contents = (Path(__file__).parent / 'app.py').read_text()
            contents = re.sub('theme=gr.themes.Default\\(\\)', f"theme='{space_id}'", contents)
            contents = re.sub('{THEME}', theme_name or repo_name, contents)
            contents = re.sub('{AUTHOR}', org_name or author, contents)
            contents = re.sub('{SPACE_NAME}', repo_name, contents)
            app_file.write(contents)
        operations = [CommitOperationAdd(path_in_repo=f'themes/theme_schema@{version}.json', path_or_fileobj=css_file.name), CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=readme_file.name), CommitOperationAdd(path_in_repo='app.py', path_or_fileobj=app_file.name)]
        huggingface_hub.create_repo(space_id, repo_type='space', space_sdk='gradio', token=hf_token, exist_ok=True, private=private)
        api.create_commit(repo_id=space_id, commit_message='Updating theme', repo_type='space', operations=operations, token=hf_token)
        url = f'https://huggingface.co/spaces/{space_id}'
        print(f'See your theme here! {url}')
        return url