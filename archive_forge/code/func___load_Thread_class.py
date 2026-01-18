from winappdbg import win32
def __load_Thread_class(self):
    global Thread
    if Thread is None:
        from winappdbg.thread import Thread