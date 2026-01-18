import re
def __read_query_alignment(record, line):
    m = __regex['start'].search(line)
    if m:
        record.query_start = int(m.group(1))
    m = __regex['align'].match(line)
    assert m is not None, 'invalid match'
    record.query_aln += m.group(1)