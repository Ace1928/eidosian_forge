import os as _os
from streamlit import logger as _logger
from streamlit import config as _config
from streamlit.deprecation_util import deprecate_func_name as _deprecate_func_name
from streamlit.version import STREAMLIT_VERSION_STRING as _STREAMLIT_VERSION_STRING
from streamlit.delta_generator import (
from streamlit.runtime.caching import (
from streamlit.runtime.connection_factory import (
from streamlit.runtime.fragment import fragment as _fragment
from streamlit.runtime.metrics_util import gather_metrics as _gather_metrics
from streamlit.runtime.secrets import secrets_singleton as _secrets_singleton
from streamlit.runtime.state import (
from streamlit.user_info import UserInfoProxy as _UserInfoProxy
from streamlit.commands.experimental_query_params import (
import streamlit.column_config as _column_config
from streamlit.echo import echo as echo
from streamlit.runtime.legacy_caching import cache as _cache
from streamlit.elements.spinner import spinner as spinner
from streamlit.commands.page_config import set_page_config as set_page_config
from streamlit.commands.execution_control import (
def _update_logger() -> None:
    _logger.set_log_level(_config.get_option('logger.level').upper())
    _logger.update_formatter()
    _logger.init_tornado_logs()