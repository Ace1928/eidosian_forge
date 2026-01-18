import re
from hacking import core
from neutron_lib.hacking import translation_checks
def _check_imports(regex, submatch, logical_line):
    m = re.match(regex, logical_line)
    if m and m.group(1) == submatch:
        return True