from twisted.python import log
from twisted.words.xish import xpath
def _addObserver(self, onetime, event, observerfn, priority, *args, **kwargs):
    if self._dispatchDepth > 0:
        self._updateQueue.append(lambda: self._addObserver(onetime, event, observerfn, priority, *args, **kwargs))
        return
    event, observers = self._getEventAndObservers(event)
    if priority not in observers:
        cbl = CallbackList()
        observers[priority] = {event: cbl}
    else:
        priorityObservers = observers[priority]
        if event not in priorityObservers:
            cbl = CallbackList()
            observers[priority][event] = cbl
        else:
            cbl = priorityObservers[event]
    cbl.addCallback(onetime, observerfn, *args, **kwargs)