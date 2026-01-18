import collections
import copy
import json
import re
import shutil
import tempfile
def _get_code(input_file):
    """Loads the ipynb file and returns a list of CodeLines."""
    raw_code = []
    with open(input_file) as in_file:
        notebook = json.load(in_file)
    cell_index = 0
    for cell in notebook['cells']:
        if is_python(cell):
            cell_lines = cell['source']
            is_line_split = False
            for line_idx, code_line in enumerate(cell_lines):
                if skip_magic(code_line, ['%', '!', '?']) or is_line_split:
                    code_line = '###!!!' + code_line
                    is_line_split = check_line_split(code_line)
                if is_line_split:
                    is_line_split = check_line_split(code_line)
                if line_idx == len(cell_lines) - 1 and code_line.endswith('\n'):
                    code_line = code_line.replace('\n', '###===')
                raw_code.append(CodeLine(cell_index, code_line.rstrip().replace('\n', '###===')))
            cell_index += 1
    return (raw_code, notebook)