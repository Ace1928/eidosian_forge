import os
import sys
import time
from rdkit import RDConfig
Context manager which uses low-level file descriptors to suppress
  output to stdout/stderr, optionally redirecting to the named file(s).

  Suppress all output
  with Silence():
    <code>

  Redirect stdout to file
  with OutputRedirectC(stdout='output.txt', mode='w'):
    <code>

  Redirect stderr to file
  with OutputRedirectC(stderr='output.txt', mode='a'):
    <code>
  http://code.activestate.com/recipes/577564-context-manager-for-low-level-redirection-of-stdou/
  >>>

  