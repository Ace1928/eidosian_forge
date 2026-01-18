from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class WaitChainNodeInfo(object):
    """
    Represents a node in the wait chain.

    It's a wrapper on the L{WAITCHAIN_NODE_INFO} structure.

    The following members are defined only
    if the node is of L{WctThreadType} type:
     - C{ProcessId}
     - C{ThreadId}
     - C{WaitTime}
     - C{ContextSwitches}

    @see: L{GetThreadWaitChain}

    @type ObjectName: unicode
    @ivar ObjectName: Object name. May be an empty string.

    @type ObjectType: int
    @ivar ObjectType: Object type.
        Should be one of the following values:
         - L{WctCriticalSectionType}
         - L{WctSendMessageType}
         - L{WctMutexType}
         - L{WctAlpcType}
         - L{WctComType}
         - L{WctThreadWaitType}
         - L{WctProcessWaitType}
         - L{WctThreadType}
         - L{WctComActivationType}
         - L{WctUnknownType}

    @type ObjectStatus: int
    @ivar ObjectStatus: Wait status.
        Should be one of the following values:
         - L{WctStatusNoAccess} I{(ACCESS_DENIED for this object)}
         - L{WctStatusRunning} I{(Thread status)}
         - L{WctStatusBlocked} I{(Thread status)}
         - L{WctStatusPidOnly} I{(Thread status)}
         - L{WctStatusPidOnlyRpcss} I{(Thread status)}
         - L{WctStatusOwned} I{(Dispatcher object status)}
         - L{WctStatusNotOwned} I{(Dispatcher object status)}
         - L{WctStatusAbandoned} I{(Dispatcher object status)}
         - L{WctStatusUnknown} I{(All objects)}
         - L{WctStatusError} I{(All objects)}

    @type ProcessId: int
    @ivar ProcessId: Process global ID.

    @type ThreadId: int
    @ivar ThreadId: Thread global ID.

    @type WaitTime: int
    @ivar WaitTime: Wait time.

    @type ContextSwitches: int
    @ivar ContextSwitches: Number of context switches.
    """

    def __init__(self, aStructure):
        self.ObjectType = aStructure.ObjectType
        self.ObjectStatus = aStructure.ObjectStatus
        if self.ObjectType == WctThreadType:
            self.ProcessId = aStructure.u.ThreadObject.ProcessId
            self.ThreadId = aStructure.u.ThreadObject.ThreadId
            self.WaitTime = aStructure.u.ThreadObject.WaitTime
            self.ContextSwitches = aStructure.u.ThreadObject.ContextSwitches
            self.ObjectName = u''
        else:
            self.ObjectName = aStructure.u.LockObject.ObjectName.value