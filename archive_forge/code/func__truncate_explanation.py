from typing import List
from typing import Optional
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.nodes import Item
def _truncate_explanation(input_lines: List[str], max_lines: Optional[int]=None, max_chars: Optional[int]=None) -> List[str]:
    """Truncate given list of strings that makes up the assertion explanation.

    Truncates to either 8 lines, or 640 characters - whichever the input reaches
    first, taking the truncation explanation into account. The remaining lines
    will be replaced by a usage message.
    """
    if max_lines is None:
        max_lines = DEFAULT_MAX_LINES
    if max_chars is None:
        max_chars = DEFAULT_MAX_CHARS
    input_char_count = len(''.join(input_lines))
    tolerable_max_chars = max_chars + 70
    tolerable_max_lines = max_lines + 2
    if len(input_lines) <= tolerable_max_lines and input_char_count <= tolerable_max_chars:
        return input_lines
    truncated_explanation = input_lines[:max_lines]
    truncated_char = True
    if len(''.join(truncated_explanation)) > tolerable_max_chars:
        truncated_explanation = _truncate_by_char_count(truncated_explanation, max_chars)
    else:
        truncated_char = False
    truncated_line_count = len(input_lines) - len(truncated_explanation)
    if truncated_explanation[-1]:
        truncated_explanation[-1] = truncated_explanation[-1] + '...'
        if truncated_char:
            truncated_line_count += 1
    else:
        truncated_explanation[-1] = '...'
    return [*truncated_explanation, '', f'...Full output truncated ({truncated_line_count} line{('' if truncated_line_count == 1 else 's')} hidden), {USAGE_MSG}']