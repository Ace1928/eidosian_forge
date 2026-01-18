from __future__ import annotations
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any, Final
from streamlit import (
from streamlit.config import CONFIG_FILENAMES
from streamlit.git_util import MIN_GIT_VERSION, GitRepo
from streamlit.logger import get_logger
from streamlit.source_util import invalidate_pages_cache
from streamlit.watcher import report_watchdog_availability, watch_dir, watch_file
from streamlit.web.server import Server, server_address_is_unix_socket, server_util
def _print_url(is_running_hello: bool) -> None:
    if is_running_hello:
        title_message = 'Welcome to Streamlit. Check out our demo in your browser.'
    else:
        title_message = 'You can now view your Streamlit app in your browser.'
    named_urls = []
    if config.is_manually_set('browser.serverAddress'):
        named_urls = [('URL', server_util.get_url(config.get_option('browser.serverAddress')))]
    elif config.is_manually_set('server.address') and (not server_address_is_unix_socket()):
        named_urls = [('URL', server_util.get_url(config.get_option('server.address')))]
    elif server_address_is_unix_socket():
        named_urls = [('Unix Socket', config.get_option('server.address'))]
    elif config.get_option('server.headless'):
        internal_ip = net_util.get_internal_ip()
        if internal_ip:
            named_urls.append(('Network URL', server_util.get_url(internal_ip)))
        external_ip = net_util.get_external_ip()
        if external_ip:
            named_urls.append(('External URL', server_util.get_url(external_ip)))
    else:
        named_urls = [('Local URL', server_util.get_url('localhost'))]
        internal_ip = net_util.get_internal_ip()
        if internal_ip:
            named_urls.append(('Network URL', server_util.get_url(internal_ip)))
    cli_util.print_to_cli('')
    cli_util.print_to_cli('  %s' % title_message, fg='blue', bold=True)
    cli_util.print_to_cli('')
    for url_name, url in named_urls:
        cli_util.print_to_cli(f'  {url_name}: ', nl=False, fg='blue')
        cli_util.print_to_cli(url, bold=True)
    cli_util.print_to_cli('')
    if is_running_hello:
        cli_util.print_to_cli('  Ready to create your own Python apps super quickly?')
        cli_util.print_to_cli('  Head over to ', nl=False)
        cli_util.print_to_cli('https://docs.streamlit.io', bold=True)
        cli_util.print_to_cli('')
        cli_util.print_to_cli('  May you create awesome apps!')
        cli_util.print_to_cli('')
        cli_util.print_to_cli('')