import sys
def __enterboxRestore(event):
    global entryWidget
    entryWidget.delete(0, len(entryWidget.get()))
    entryWidget.insert(0, __enterboxDefaultText)