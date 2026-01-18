import re
def __read_lengths(record, line):
    if not line.startswith('length1='):
        raise ValueError(f"Line does not start with 'length1=':\n{line}")
    m = __regex['lengths'].search(line)
    record.query_length = int(m.group(1))
    record.query_filtered_length = float(m.group(2))
    record.hit_length = int(m.group(3))
    record.hit_filtered_length = float(m.group(4))