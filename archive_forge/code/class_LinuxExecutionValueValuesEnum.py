from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinuxExecutionValueValuesEnum(_messages.Enum):
    """Defines how Linux actions are allowed to execute. DO NOT USE:
    Experimental / unlaunched feature.

    Values:
      LINUX_EXECUTION_UNSPECIFIED: Default value, if not explicitly set.
        Equivalent to FORBIDDEN.
      LINUX_EXECUTION_FORBIDDEN: Linux actions and worker pools are forbidden.
      LINUX_EXECUTION_UNRESTRICTED: No restrictions on execution of Linux
        actions.
      LINUX_EXECUTION_HARDENED_GVISOR: Linux actions will be hardened using
        gVisor. Actions that specify a configuration incompatible with gVisor
        hardening will be rejected. Example per-action platform properties
        that are incompatible with gVisor hardening are: 1. dockerRuntime is
        set to a value other than "runsc". Leaving dockerRuntime unspecified
        *is* compatible with gVisor. 2. dockerPrivileged is set to "true".
        etc.
      LINUX_EXECUTION_HARDENED_GVISOR_OR_TERMINAL: Linux actions will be
        hardened using gVisor if their configuration is compatible with gVisor
        hardening. Otherwise, the action will be terminal, i.e., the worker VM
        that runs the action will be terminated after the action finishes.
    """
    LINUX_EXECUTION_UNSPECIFIED = 0
    LINUX_EXECUTION_FORBIDDEN = 1
    LINUX_EXECUTION_UNRESTRICTED = 2
    LINUX_EXECUTION_HARDENED_GVISOR = 3
    LINUX_EXECUTION_HARDENED_GVISOR_OR_TERMINAL = 4