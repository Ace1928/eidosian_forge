import urllib.parse
from typing import Callable, Dict, Optional, Union
import wandb
from wandb import env
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps.utils import handle_sweep_config_violations
from . import wandb_login
def _get_sweep_url(api, sweep_id):
    """Return sweep url if we can figure it out."""
    if api.api_key:
        if api.settings('entity') is None:
            viewer = api.viewer()
            if viewer.get('entity'):
                api.set_setting('entity', viewer['entity'])
        project = api.settings('project')
        if not project:
            return
        if api.settings('entity'):
            return '{base}/{entity}/{project}/sweeps/{sweepid}'.format(base=api.app_url, entity=urllib.parse.quote(api.settings('entity')), project=urllib.parse.quote(project), sweepid=urllib.parse.quote(sweep_id))