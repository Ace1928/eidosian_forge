import base64
import typing
def _pem_lines(contents: bytes, pem_start: bytes, pem_end: bytes) -> typing.Iterator[bytes]:
    """Generator over PEM lines between pem_start and pem_end."""
    in_pem_part = False
    seen_pem_start = False
    for line in contents.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == pem_start:
            if in_pem_part:
                raise ValueError('Seen start marker "%r" twice' % pem_start)
            in_pem_part = True
            seen_pem_start = True
            continue
        if not in_pem_part:
            continue
        if in_pem_part and line == pem_end:
            in_pem_part = False
            break
        if b':' in line:
            continue
        yield line
    if not seen_pem_start:
        raise ValueError('No PEM start marker "%r" found' % pem_start)
    if in_pem_part:
        raise ValueError('No PEM end marker "%r" found' % pem_end)