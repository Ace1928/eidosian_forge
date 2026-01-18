from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OsInfo(_messages.Message):
    """Operating system information for the VM.

  Fields:
    architecture: The system architecture of the operating system.
    hostname: The VM hostname.
    kernelRelease: The kernel release of the operating system.
    kernelVersion: The kernel version of the operating system.
    longName: The operating system long name. For example 'Debian GNU/Linux 9'
      or 'Microsoft Window Server 2019 Datacenter'.
    osconfigAgentVersion: The current version of the OS Config agent running
      on the VM.
    shortName: The operating system short name. For example, 'windows' or
      'debian'.
    version: The version of the operating system.
  """
    architecture = _messages.StringField(1)
    hostname = _messages.StringField(2)
    kernelRelease = _messages.StringField(3)
    kernelVersion = _messages.StringField(4)
    longName = _messages.StringField(5)
    osconfigAgentVersion = _messages.StringField(6)
    shortName = _messages.StringField(7)
    version = _messages.StringField(8)