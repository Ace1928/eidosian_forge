from .arc import EllipticalArc
import re
def _tokenize_path(pathdef):
    arc_cmd = None
    for x in COMMAND_RE.split(pathdef):
        if x in COMMANDS:
            arc_cmd = x if x in ARC_COMMANDS else None
            yield x
            continue
        if arc_cmd:
            try:
                yield from _tokenize_arc_arguments(x)
            except ValueError as e:
                raise ValueError(f"Invalid arc command: '{arc_cmd}{x}'") from e
        else:
            for token in FLOAT_RE.findall(x):
                yield token