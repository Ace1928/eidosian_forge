import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _build_re_values():
    quoted_re = '\n                    "      # open quote followed by zero or more of:\n                    (?:\n                        (?<!\\\\)    # no additional backslash\n                        (?:\\\\\\\\)*  # maybe escaped backslashes\n                        \\\\"        # escaped quote\n                    |\n                        \\\\[^"]     # escaping a non-quote\n                    |\n                        [^"\\\\]     # non-quote char\n                    )*\n                    "      # close quote\n                    '
    value_re = '(?:\n        %s|          # a value may be surrounded by "\n        %s|          # or by \'\n        [^,\\s"\'{}]+  # or may contain no characters requiring quoting\n        )' % (quoted_re, quoted_re.replace('"', "'"))
    dense = re.compile("(?x)\n        ,                # may follow ','\n        \\s*\n        ((?=,)|$|{value_re})  # empty or value\n        |\n        (\\S.*)           # error\n        ".format(value_re=value_re))
    sparse = re.compile("(?x)\n        (?:^\\s*\\{|,)   # may follow ',', or '{' at line start\n        \\s*\n        (\\d+)          # attribute key\n        \\s+\n        (%(value_re)s) # value\n        |\n        (?!}\\s*$)      # not an error if it's }$\n        (?!^\\s*{\\s*}\\s*$)  # not an error if it's ^{}$\n        \\S.*           # error\n        " % {'value_re': value_re})
    return (dense, sparse)