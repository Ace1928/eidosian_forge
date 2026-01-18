from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JupyterConfig(_messages.Message):
    """Jupyter configuration for an interactive session.

  Enums:
    KernelValueValuesEnum: Optional. Kernel

  Fields:
    displayName: Optional. Display name, shown in the Jupyter kernelspec card.
    kernel: Optional. Kernel
  """

    class KernelValueValuesEnum(_messages.Enum):
        """Optional. Kernel

    Values:
      KERNEL_UNSPECIFIED: The kernel is unknown.
      PYTHON: Python kernel.
      SCALA: Scala kernel.
    """
        KERNEL_UNSPECIFIED = 0
        PYTHON = 1
        SCALA = 2
    displayName = _messages.StringField(1)
    kernel = _messages.EnumField('KernelValueValuesEnum', 2)