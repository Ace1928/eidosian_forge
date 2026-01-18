import re
import numpy
class ReFile:
    """convenience class for dealing with files with comments

  blank (all whitespace) lines, and lines beginning with comment
    characters are skipped.

  anything following a comment character on a line is stripped off
  """

    def readline(self):
        """ read the next line and return it.

    return '' on EOF

    """
        result = ''
        while result == '':
            inLine = self.inFile.readline()
            if inLine == '':
                return ''
            result = self.regExp.split(inLine)[0].strip()
        return result

    def readlines(self):
        """ return a list of all the lines left in the file

    return [] if there are none

    """
        res = []
        inLines = self.inFile.readlines()
        for line in inLines:
            result = self.regExp.split(line)[0].strip()
            if result != '':
                res.append(result)
        return res

    def rewind(self):
        """ rewinds the file (seeks to the beginning)

    """
        self.inFile.seek(0)

    def __init__(self, fileName, mode='r', comment='#', trailer='\\n'):
        if trailer is not None and trailer != '':
            comment = comment + '|' + trailer
        self.regExp = re.compile(comment)
        self.inFile = open(fileName, mode)