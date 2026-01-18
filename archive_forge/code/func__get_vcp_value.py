import re
from .datetime_helpers import (
from .dom_helpers import get_children
def _get_vcp_value(el):
    if 'value-title' in el.get('class', []):
        return el.get('title')
    return el.get_text()