from typing import Any, Dict, Iterable
def _num_lines_from_num_bytes(file, num_bytes):
    size = 0
    num_lines = 0
    content = file['content']
    while num_lines < len(content):
        size += _str_size(content[num_lines])
        if size > num_bytes:
            break
        num_lines += 1
    return num_lines