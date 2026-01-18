import ast
from typing import Optional, Union
def _format_message(self) -> str:
    node = self.node
    if self.src is None:
        source_excerpt = ' <source unavailable>'
    else:
        source_excerpt = self.src.split('\n')[:node.lineno][-self.source_line_count_max_in_message:]
        if source_excerpt:
            source_excerpt.append(' ' * node.col_offset + '^')
            source_excerpt = '\n'.join(source_excerpt)
        else:
            source_excerpt = ' <source empty>'
    message = 'at {}:{}:{}'.format(node.lineno, node.col_offset, source_excerpt)
    if self.error_message:
        message += '\n' + self.error_message
    return message