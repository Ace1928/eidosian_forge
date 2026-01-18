import re
from .datetime_helpers import (
from .dom_helpers import get_children
def _get_vcp_children(el):
    return [c for c in get_children(el) if c.has_attr('class') and ('value' in c['class'] or 'value-title' in c['class'])]