import re
from Bio import File
def _parse_remark_465(line):
    """Parse missing residue remarks.

    Returns a dictionary describing the missing residue.
    The specification for REMARK 465 at
    http://www.wwpdb.org/documentation/file-format-content/format33/remarks2.html#REMARK%20465
    only gives templates, but does not say they have to be followed.
    So we assume that not all pdb-files with a REMARK 465 can be understood.

    Returns a dictionary with the following keys:
    "model", "res_name", "chain", "ssseq", "insertion"
    """
    if line:
        assert line[0] != ' ' and line[-1] not in '\n ', 'line has to be stripped'
    pattern = re.compile('\n        (\\d+\\s[\\sA-Z][\\sA-Z][A-Z] |   # Either model number + residue name\n            [A-Z]{1,3})               # Or only residue name with 1 (RNA) to 3 letters\n        \\s ([A-Za-z0-9])              # A single character chain\n        \\s+(-?\\d+[A-Za-z]?)$          # Residue number: A digit followed by an optional\n                                      # insertion code (Hetero-flags make no sense in\n                                      # context with missing res)\n        ', re.VERBOSE)
    match = pattern.match(line)
    if match is None:
        return None
    residue = {}
    if ' ' in match.group(1):
        model, residue['res_name'] = match.group(1).split()
        residue['model'] = int(model)
    else:
        residue['model'] = None
        residue['res_name'] = match.group(1)
    residue['chain'] = match.group(2)
    try:
        residue['ssseq'] = int(match.group(3))
    except ValueError:
        residue['insertion'] = match.group(3)[-1]
        residue['ssseq'] = int(match.group(3)[:-1])
    else:
        residue['insertion'] = None
    return residue