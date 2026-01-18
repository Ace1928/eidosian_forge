import re
def __read_threshold(record, line):
    if not line.startswith('Threshold'):
        raise ValueError(f"Line does not start with 'Threshold':\n{line}")
    m = __regex['threshold'].search(line)
    record.gap_threshold = float(m.group(1))