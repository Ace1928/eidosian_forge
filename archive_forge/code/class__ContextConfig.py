from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
from boto import config
import gslib
from gslib import exception
from gslib.utils import boto_util
from gslib.utils import execution_util
class _ContextConfig(object):
    """Represents the configurations associated with context aware access.

  Only one instance of Config can be created for the program.
  """

    def __init__(self, logger):
        """Initializes config.

    Args:
      logger (logging.logger): gsutil logger.
    """
        self.logger = logger
        self.use_client_certificate = config.getbool('Credentials', 'use_client_certificate')
        self.client_cert_path = None
        if not self.use_client_certificate:
            return
        atexit.register(self._unprovision_client_cert)
        self.client_cert_path = os.path.join(boto_util.GetGsutilStateDir(), 'caa_cert.pem')
        try:
            self._provision_client_cert(self.client_cert_path)
        except CertProvisionError as e:
            self.logger.error('Failed to provision client certificate: %s' % e)

    def _provision_client_cert(self, cert_path):
        """Executes certificate provider to obtain client certificate and keys."""
        cert_command_string = config.get('Credentials', 'cert_provider_command', None)
        if cert_command_string:
            cert_command = cert_command_string.split(' ')
        else:
            cert_command = _default_command()
        try:
            command_stdout_string, _ = execution_util.ExecuteExternalCommand(cert_command)
            sections = _split_pem_into_sections(command_stdout_string, self.logger)
            with open(cert_path, 'w+') as f:
                f.write(sections['CERTIFICATE'])
                if 'ENCRYPTED PRIVATE KEY' in sections:
                    f.write(sections['ENCRYPTED PRIVATE KEY'])
                    self.client_cert_password = sections['PASSPHRASE'].splitlines()[1]
                else:
                    f.write(sections['PRIVATE KEY'])
                    self.client_cert_password = None
        except (exception.ExternalBinaryError, OSError) as e:
            raise CertProvisionError(e)
        except KeyError as e:
            raise CertProvisionError('Invalid output format from certificate provider, no %s' % e)

    def _unprovision_client_cert(self):
        """Cleans up any files or resources provisioned during config init."""
        if self.client_cert_path is not None:
            try:
                os.remove(self.client_cert_path)
                self.logger.debug('Unprovisioned client cert: %s' % self.client_cert_path)
            except OSError as e:
                self.logger.error('Failed to remove client certificate: %s' % e)