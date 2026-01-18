from __future__ import annotations
from typing import Final
from urllib.parse import urljoin
import tornado.web
from streamlit import config, net_util, url_util
def _get_server_address_if_manually_set() -> str | None:
    if config.is_manually_set('browser.serverAddress'):
        return url_util.get_hostname(config.get_option('browser.serverAddress'))
    return None