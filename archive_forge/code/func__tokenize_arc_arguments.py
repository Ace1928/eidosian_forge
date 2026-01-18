from .arc import EllipticalArc
import re
def _tokenize_arc_arguments(arcdef):
    raw_args = [s for s in SEPARATOR_RE.split(arcdef) if s]
    if not raw_args:
        raise ValueError(f"Not enough arguments: '{arcdef}'")
    raw_args.reverse()
    i = 0
    while raw_args:
        arg = raw_args.pop()
        name, pattern = ARC_ARGUMENT_TYPES[i]
        match = pattern.search(arg)
        if not match:
            raise ValueError(f"Invalid argument for '{name}' parameter: {arg!r}")
        j, k = match.span()
        yield arg[j:k]
        arg = arg[k:]
        if arg:
            raw_args.append(arg)
        if i == 6:
            i = 0
        else:
            i += 1
    if i != 0:
        raise ValueError(f"Not enough arguments: '{arcdef}'")