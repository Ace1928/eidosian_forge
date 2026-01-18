import re
def _break_line_iterator(line, line_length):
    for i in range(0, len(line), line_length):
        yield line[i:i + line_length]