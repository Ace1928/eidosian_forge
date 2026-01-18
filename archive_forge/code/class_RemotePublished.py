import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
class RemotePublished(flavors.RemoteCache):
    """The local representation of remote Publishable object."""
    isActivated = 0
    _wasCleanWhenLoaded = 0

    def getFileName(self, ext='pub'):
        return '{}-{}-{}.{}'.format(self.service, self.perspective, str(self.publishedID), ext)

    def setCopyableState(self, state):
        self.__dict__.update(state)
        self._activationListeners = []
        try:
            with open(self.getFileName(), 'rb') as dataFile:
                data = dataFile.read()
        except OSError:
            recent = 0
        else:
            newself = jelly.unjelly(banana.decode(data))
            recent = newself.timestamp == self.timestamp
        if recent:
            self._cbGotUpdate(newself.__dict__)
            self._wasCleanWhenLoaded = 1
        else:
            self.remote.callRemote('getStateToPublish').addCallbacks(self._cbGotUpdate)

    def __getstate__(self):
        other = self.__dict__.copy()
        del other['broker']
        del other['remote']
        del other['luid']
        del other['_activationListeners']
        del other['isActivated']
        return other

    def _cbGotUpdate(self, newState):
        self.__dict__.update(newState)
        self.isActivated = 1
        for listener in self._activationListeners:
            listener(self)
        self._activationListeners = []
        self.activated()
        with open(self.getFileName(), 'wb') as dataFile:
            dataFile.write(banana.encode(jelly.jelly(self)))

    def activated(self):
        """Implement this method if you want to be notified when your
        publishable subclass is activated.
        """

    def callWhenActivated(self, callback):
        """Externally register for notification when this publishable has received all relevant data."""
        if self.isActivated:
            callback(self)
        else:
            self._activationListeners.append(callback)