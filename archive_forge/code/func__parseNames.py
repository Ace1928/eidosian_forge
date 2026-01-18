from os import getpid
from typing import Dict, List, Mapping, Optional, Sequence
from attrs import Factory, define
def _parseNames(environ: Mapping[str, str]) -> Sequence[str]:
    """
    Parse the I{LISTEN_FDNAMES} environment variable supplied by systemd.

    @param environ: The environment variable mapping in which to look for the
        value to parse.

    @return: The names of the inherited descriptors, in order.
    """
    names = environ.get('LISTEN_FDNAMES', '')
    if len(names) > 0:
        return tuple(names.split(':'))
    return ()