import sys
def _normalize_docstring(docstring):
    """Normalizes the docstring.

  Replaces tabs with spaces, removes leading and trailing blanks lines, and
  removes any indentation.

  Copied from PEP-257:
  https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation

  Args:
    docstring: the docstring to normalize

  Returns:
    The normalized docstring
  """
    if not docstring:
        return ''
    lines = docstring.expandtabs().splitlines()
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    while trimmed and (not trimmed[-1]):
        trimmed.pop()
    while trimmed and (not trimmed[0]):
        trimmed.pop(0)
    return '\n'.join(trimmed)