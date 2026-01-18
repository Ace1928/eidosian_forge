import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
class IconData(IniFile):
    """Class to parse and validate IconData Files"""

    def __init__(self):
        IniFile.__init__(self)

    def __repr__(self):
        displayname = self.getDisplayName()
        if displayname:
            return '<IconData: %s>' % displayname
        else:
            return '<IconData>'

    def parse(self, file):
        IniFile.parse(self, file, ['Icon Data'])

    def getDisplayName(self):
        """Retrieve the display name from the icon data, if one is specified."""
        return self.get('DisplayName', locale=True)

    def getEmbeddedTextRectangle(self):
        """Retrieve the embedded text rectangle from the icon data as a list of
        numbers (x0, y0, x1, y1), if it is specified."""
        return self.get('EmbeddedTextRectangle', type='integer', list=True)

    def getAttachPoints(self):
        """Retrieve the anchor points for overlays & emblems from the icon data,
        as a list of co-ordinate pairs, if they are specified."""
        return self.get('AttachPoints', type='point', list=True)

    def checkExtras(self):
        if self.fileExtension != '.icon':
            self.warnings.append('Unknown File extension')

    def checkGroup(self, group):
        if not (group == self.defaultGroup or (re.match('^\\[X-', group) and is_ascii(group))):
            self.errors.append('Invalid Group name: %s' % group.encode('ascii', 'replace'))

    def checkKey(self, key, value, group):
        if re.match('^DisplayName' + xdg.Locale.regex + '$', key):
            pass
        elif key == 'EmbeddedTextRectangle':
            self.checkValue(key, value, type='integer', list=True)
        elif key == 'AttachPoints':
            self.checkValue(key, value, type='point', list=True)
        elif re.match('^X-[a-zA-Z0-9-]+', key):
            pass
        else:
            self.errors.append('Invalid key: %s' % key)