from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
class JupyterConfigFactory(object):
    """Factory for JupyterConfig message.

  Factory to add JupyterConfig message arguments to argument parser and create
  JupyterConfig message from parsed arguments.
  """

    def __init__(self, dataproc):
        """Factory for JupyterConfig message.

    Args:
      dataproc: A api_lib.dataproc.Dataproc instance.
    """
        self.dataproc = dataproc

    def GetMessage(self, args):
        """Builds a JupyterConfig message according to user settings.

    Args:
      args: Parsed arguments.

    Returns:
      JupyterConfig: A JupyterConfig message instance.
    """
        jupyter_config = self.dataproc.messages.JupyterConfig()
        if args.kernel:
            jupyter_config.kernel = arg_utils.ChoiceToEnum(args.kernel, self.dataproc.messages.JupyterConfig.KernelValueValuesEnum)
        return jupyter_config