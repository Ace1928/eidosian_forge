import pythoncom
from win32com.shell import shell, shellcon
from win32com.server.policy import DesignatedWrapPolicy
def PostDeleteItem(self, flags, item, hr_delete, newly_created):
    if newly_created:
        self.newItem = newly_created.GetDisplayName(shellcon.SHGDN_FORPARSING)