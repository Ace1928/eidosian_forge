import enum
import os
import sys
from typing import Dict, Optional, Tuple
import click
import wandb
from wandb.errors import AuthenticationError, UsageError
from wandb.old.settings import Settings as OldSettings
from ..apis import InternalApi
from .internal.internal_api import Api
from .lib import apikey
from .wandb_settings import Settings, Source
class _WandbLogin:

    def __init__(self):
        self.kwargs: Optional[Dict] = None
        self._settings: Optional[Settings] = None
        self._backend = None
        self._silent = None
        self._entity = None
        self._wl = None
        self._key = None
        self._relogin = None

    def setup(self, kwargs):
        self.kwargs = kwargs
        login_settings: Settings = wandb.Settings()
        settings_param = kwargs.pop('_settings', None)
        if settings_param is not None:
            if isinstance(settings_param, Settings):
                login_settings._apply_settings(settings_param)
            elif isinstance(settings_param, dict):
                login_settings.update(settings_param, source=Source.LOGIN)
        _logger = wandb.setup()._get_logger()
        self._relogin = kwargs.pop('relogin', None)
        login_settings._apply_login(kwargs, _logger=_logger)
        self._wl = wandb.setup(settings=login_settings)
        self._settings = self._wl.settings

    def is_apikey_configured(self):
        return apikey.api_key(settings=self._settings) is not None

    def set_backend(self, backend):
        self._backend = backend

    def set_silent(self, silent: bool):
        self._silent = silent

    def set_entity(self, entity: str):
        self._entity = entity

    def login(self):
        apikey_configured = self.is_apikey_configured()
        if self._settings.relogin or self._relogin:
            apikey_configured = False
        if not apikey_configured:
            return False
        if not self._silent:
            self.login_display()
        return apikey_configured

    def login_display(self):
        username = self._wl._get_username()
        if username:
            entity = self._entity or self._wl._get_entity()
            entity_str = ''
            if entity and entity in self._wl._get_teams() and (entity != username):
                entity_str = f' ({click.style(entity, fg='yellow')})'
            login_state_str = f'Currently logged in as: {click.style(username, fg='yellow')}{entity_str}'
        else:
            login_state_str = 'W&B API key is configured'
        login_info_str = f'Use {click.style('`wandb login --relogin`', bold=True)} to force relogin'
        wandb.termlog(f'{login_state_str}. {login_info_str}', repeat=False)

    def configure_api_key(self, key):
        if self._settings._notebook and (not self._settings.silent):
            wandb.termwarn("If you're specifying your api key in code, ensure this code is not shared publicly.\nConsider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.")
        apikey.write_key(self._settings, key)
        self.update_session(key)
        self._key = key

    def update_session(self, key: Optional[str], status: ApiKeyStatus=ApiKeyStatus.VALID) -> None:
        _logger = wandb.setup()._get_logger()
        login_settings = dict()
        if status == ApiKeyStatus.OFFLINE:
            login_settings = dict(mode='offline')
        elif status == ApiKeyStatus.DISABLED:
            login_settings = dict(mode='disabled')
        elif key:
            login_settings = dict(api_key=key)
        self._wl._settings._apply_login(login_settings, _logger=_logger)
        if not self._wl.settings._offline:
            self._wl._update_user_settings()

    def _prompt_api_key(self) -> Tuple[Optional[str], ApiKeyStatus]:
        api = Api(self._settings)
        while True:
            try:
                key = apikey.prompt_api_key(self._settings, api=api, no_offline=self._settings.force if self._settings else None, no_create=self._settings.force if self._settings else None)
            except ValueError as e:
                wandb.termerror(e.args[0])
                continue
            except TimeoutError:
                wandb.termlog('W&B disabled due to login timeout.')
                return (None, ApiKeyStatus.DISABLED)
            if key is False:
                return (None, ApiKeyStatus.NOTTY)
            if not key:
                return (None, ApiKeyStatus.OFFLINE)
            return (key, ApiKeyStatus.VALID)

    def prompt_api_key(self):
        key, status = self._prompt_api_key()
        if status == ApiKeyStatus.NOTTY:
            directive = 'wandb login [your_api_key]' if self._settings._cli_only_mode else 'wandb.login(key=[your_api_key])'
            raise UsageError('api_key not configured (no-tty). call ' + directive)
        self.update_session(key, status=status)
        self._key = key

    def propogate_login(self):
        if self._backend:
            pass