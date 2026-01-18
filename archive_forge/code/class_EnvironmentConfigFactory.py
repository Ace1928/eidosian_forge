from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.shared_messages import (
from googlecloudsdk.command_lib.dataproc.shared_messages import (
class EnvironmentConfigFactory(object):
    """Factory for EnvironmentConfig message.

  Add arguments related to EnvironmentConfig to argument parser and create
  EnvironmentConfig message from parsed arguments.
  """

    def __init__(self, dataproc, execution_config_factory_override=None, peripherals_config_factory_override=None):
        """Factory for EnvironmentConfig message.

    Args:
      dataproc: A api_lib.dataproc.Dataproc instance.
      execution_config_factory_override: Override the default
      ExecutionConfigFactory instance. This is a keyword argument.
      peripherals_config_factory_override: Override the default
      PeripheralsConfigFactory instance.
    """
        self.dataproc = dataproc
        self.execution_config_factory = execution_config_factory_override
        if not self.execution_config_factory:
            self.execution_config_factory = ecf.ExecutionConfigFactory(self.dataproc)
        self.peripherals_config_factory = peripherals_config_factory_override
        if not self.peripherals_config_factory:
            self.peripherals_config_factory = pcf.PeripheralsConfigFactory(self.dataproc)

    def GetMessage(self, args):
        """Builds an EnvironmentConfig message instance.

    Args:
      args: Parsed arguments.

    Returns:
      EnvironmentConfig: An environmentConfig message instance. Returns none
      if all fields are None.
    """
        kwargs = {}
        execution_config = self.execution_config_factory.GetMessage(args)
        if execution_config:
            kwargs['executionConfig'] = execution_config
        peripherals_config = self.peripherals_config_factory.GetMessage(args)
        if peripherals_config:
            kwargs['peripheralsConfig'] = peripherals_config
        if not kwargs:
            return None
        return self.dataproc.messages.EnvironmentConfig(**kwargs)