from zope.interface import implementer
from twisted.internet.interfaces import IConsumer, IPushProducer
import pywintypes
import win32api
import win32file
import win32pipe
@implementer(IPushProducer)
class _PollableReadPipe(_PollableResource):

    def __init__(self, pipe, receivedCallback, lostCallback):
        self.pipe = pipe
        self.receivedCallback = receivedCallback
        self.lostCallback = lostCallback

    def checkWork(self):
        finished = 0
        fullDataRead = []
        while 1:
            try:
                buffer, bytesToRead, result = win32pipe.PeekNamedPipe(self.pipe, 1)
                if not bytesToRead:
                    break
                hr, data = win32file.ReadFile(self.pipe, bytesToRead, None)
                fullDataRead.append(data)
            except win32api.error:
                finished = 1
                break
        dataBuf = b''.join(fullDataRead)
        if dataBuf:
            self.receivedCallback(dataBuf)
        if finished:
            self.cleanup()
        return len(dataBuf)

    def cleanup(self):
        self.deactivate()
        self.lostCallback()

    def close(self):
        try:
            win32api.CloseHandle(self.pipe)
        except pywintypes.error:
            pass

    def stopProducing(self):
        self.close()

    def pauseProducing(self):
        self.deactivate()

    def resumeProducing(self):
        self.activate()