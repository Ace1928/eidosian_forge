import sys
import sysconfig
def _aix_bos_rte():
    """
    Return a Tuple[str, int] e.g., ['7.1.4.34', 1806]
    The fileset bos.rte represents the current AIX run-time level. It's VRMF and
    builddate reflect the current ABI levels of the runtime environment.
    If no builddate is found give a value that will satisfy pep425 related queries
    """
    out = subprocess.check_output(['/usr/bin/lslpp', '-Lqc', 'bos.rte'])
    out = out.decode('utf-8')
    out = out.strip().split(':')
    _bd = int(out[-1]) if out[-1] != '' else 9988
    return (str(out[2]), _bd)