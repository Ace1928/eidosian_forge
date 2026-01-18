import re
from yaql.language import expressions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
from yaql import yaqlization
def _match_name_to_entry(name, entry):
    if name == entry:
        return True
    elif isinstance(entry, REGEX_TYPE):
        return entry.search(name) is not None
    elif callable(entry):
        return entry(name)
    return False