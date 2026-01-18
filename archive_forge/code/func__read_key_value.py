from . import Record
def _read_key_value(line):
    words = line[1:].split('=', 1)
    try:
        key, value = words
        value = value.strip()
    except ValueError:
        key = words[0]
        value = ''
    key = key.strip()
    return (key, value)