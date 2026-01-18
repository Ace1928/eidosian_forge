from typing import Any, Dict, Tuple
from langchain.chains.query_constructor.ir import (
def _get_json_operator(self, value: Any) -> str:
    if isinstance(value, str):
        return '->>'
    else:
        return '->'