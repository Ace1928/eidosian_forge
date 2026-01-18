import sys
def __getfilesystemencoding():
    """
    Note: there's a copy of this method in interpreterInfo.py
    """
    try:
        ret = sys.getfilesystemencoding()
        if not ret:
            raise RuntimeError('Unable to get encoding.')
        return ret
    except:
        try:
            from java.lang import System
            env = System.getProperty('os.name').lower()
            if env.find('win') != -1:
                return 'ISO-8859-1'
            return 'utf-8'
        except:
            pass
        if sys.platform == 'win32':
            return 'mbcs'
        return 'utf-8'