import pythoncom
from win32com.shell import shell, shellcon
from win32com.server.policy import DesignatedWrapPolicy
def PreDeleteItem(self, flags, item):
    return 0 if flags & shellcon.TSF_DELETE_RECYCLE_IF_POSSIBLE else 2147500037