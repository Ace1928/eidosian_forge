from json import dumps, loads
import re
def _to_snake_case(n):
    return re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', n).lower()