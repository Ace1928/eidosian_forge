from typing import Callable
import requests
from urllib3.exceptions import InsecureRequestWarning
import wandb
from wandb import env, util
from .internal import Api as InternalApi  # noqa
from .public import Api as PublicApi  # noqa
def _disable_ssl() -> Callable[[], None]:
    wandb.termwarn('Disabling SSL verification.  Connections to this server are not verified and may be insecure!')
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    old_merge_environment_settings = requests.Session.merge_environment_settings

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False
        return settings
    requests.Session.merge_environment_settings = merge_environment_settings

    def reset():
        requests.Session.merge_environment_settings = old_merge_environment_settings
    return reset