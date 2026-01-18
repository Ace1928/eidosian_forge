from typing import List
from typing import Optional
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.nodes import Item
def _truncate_by_char_count(input_lines: List[str], max_chars: int) -> List[str]:
    iterated_char_count = 0
    for iterated_index, input_line in enumerate(input_lines):
        if iterated_char_count + len(input_line) > max_chars:
            break
        iterated_char_count += len(input_line)
    truncated_result = input_lines[:iterated_index]
    final_line = input_lines[iterated_index]
    if final_line:
        final_line_truncate_point = max_chars - iterated_char_count
        final_line = final_line[:final_line_truncate_point]
    truncated_result.append(final_line)
    return truncated_result