from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class ServiceStatusEntry(object):
    """
    Service status entry returned by L{EnumServicesStatus}.
    """

    def __init__(self, raw):
        """
        @type  raw: L{ENUM_SERVICE_STATUSA} or L{ENUM_SERVICE_STATUSW}
        @param raw: Raw structure for this service status entry.
        """
        self.ServiceName = raw.lpServiceName
        self.DisplayName = raw.lpDisplayName
        self.ServiceType = raw.ServiceStatus.dwServiceType
        self.CurrentState = raw.ServiceStatus.dwCurrentState
        self.ControlsAccepted = raw.ServiceStatus.dwControlsAccepted
        self.Win32ExitCode = raw.ServiceStatus.dwWin32ExitCode
        self.ServiceSpecificExitCode = raw.ServiceStatus.dwServiceSpecificExitCode
        self.CheckPoint = raw.ServiceStatus.dwCheckPoint
        self.WaitHint = raw.ServiceStatus.dwWaitHint

    def __str__(self):
        output = []
        if self.ServiceType & SERVICE_INTERACTIVE_PROCESS:
            output.append('Interactive service')
        else:
            output.append('Service')
        if self.DisplayName:
            output.append('"%s" (%s)' % (self.DisplayName, self.ServiceName))
        else:
            output.append('"%s"' % self.ServiceName)
        if self.CurrentState == SERVICE_CONTINUE_PENDING:
            output.append('is about to continue.')
        elif self.CurrentState == SERVICE_PAUSE_PENDING:
            output.append('is pausing.')
        elif self.CurrentState == SERVICE_PAUSED:
            output.append('is paused.')
        elif self.CurrentState == SERVICE_RUNNING:
            output.append('is running.')
        elif self.CurrentState == SERVICE_START_PENDING:
            output.append('is starting.')
        elif self.CurrentState == SERVICE_STOP_PENDING:
            output.append('is stopping.')
        elif self.CurrentState == SERVICE_STOPPED:
            output.append('is stopped.')
        return ' '.join(output)