import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class TextOutputArea(TextOutput):
    WRAP, TRUNCATE = range(2)

    def __init__(self, size=None, longLines=WRAP):
        TextOutput.__init__(self, size)
        self.longLines = longLines

    def render(self, width, height, terminal):
        n = 0
        inputLines = self.text.splitlines()
        outputLines = []
        while inputLines:
            if self.longLines == self.WRAP:
                line = inputLines.pop(0)
                if not isinstance(line, str):
                    line = line.decode('utf-8')
                wrappedLines = []
                for wrappedLine in tptext.greedyWrap(line, width):
                    if not isinstance(wrappedLine, bytes):
                        wrappedLine = wrappedLine.encode('utf-8')
                    wrappedLines.append(wrappedLine)
                outputLines.extend(wrappedLines or [b''])
            else:
                outputLines.append(inputLines.pop(0)[:width])
            if len(outputLines) >= height:
                break
        for n, L in enumerate(outputLines[:height]):
            terminal.cursorPosition(0, n)
            terminal.write(L)