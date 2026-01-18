from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import subprocess
import tempfile
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import log
import six
def RunOpenSSL(self, cmd_args, cmd_input=None):
    """Run an openssl command with optional input and return the output."""
    command = [self.openssl_executable]
    command.extend(cmd_args)
    try:
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, stderr = p.communicate(cmd_input)
        log.debug('Ran command "{0}" with standard error of:\n{1}'.format(' '.join(command), stderr))
    except OSError as e:
        raise OpenSSLException('[{0}] exited with [{1}].'.format(command[0], e.strerror))
    if p.returncode:
        raise OpenSSLException('[{0}] exited with return code [{1}]:\n{2}.'.format(command[0], p.returncode, stderr))
    return output