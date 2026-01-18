import re
def __read_names(record, line):
    if 'Ali1:' not in line:
        raise ValueError(f"Line does not contain 'Ali1:':\n{line}")
    m = __regex['names'].search(line)
    record.query = m.group(1)
    record.hit = m.group(2)