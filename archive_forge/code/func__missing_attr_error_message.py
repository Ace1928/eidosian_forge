from __future__ import annotations
from typing import Any, Final, Iterator, MutableMapping
from streamlit import logger as _logger
from streamlit import runtime
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.state.common import require_valid_user_key
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.runtime.state.session_state import SessionState
from streamlit.type_util import Key
def _missing_attr_error_message(attr_name: str) -> str:
    return f'st.session_state has no attribute "{attr_name}". Did you forget to initialize it? More info: https://docs.streamlit.io/library/advanced-features/session-state#initialization'