from typing import Any, Dict, Iterable
def _split_file(file, num_lines):
    offset = file['offset']
    content = file['content']
    name = file['name']
    f1 = {'offset': offset, 'content': content[:num_lines], 'name': name}
    f2 = {'offset': offset + num_lines, 'content': content[num_lines:], 'name': name}
    return (f1, f2)