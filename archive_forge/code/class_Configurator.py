from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
class Configurator(object):
    """Base configurator class.

  Configurators generate config files for specific classes of runtimes.  They
  are returned by the Fingerprint functions in the runtimes sub-package after
  a successful match of the runtime's heuristics.
  """

    def GenerateConfigs(self):
        """Generate all configuration files for the module.

    Generates config files in the current working directory.

    Returns:
      (callable()) Function that will delete all of the generated files.
    """
        raise NotImplementedError()