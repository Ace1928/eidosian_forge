import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
class Publishable(flavors.Cacheable):
    """An object whose cached state persists across sessions."""

    def __init__(self, publishedID):
        self.republish()
        self.publishedID = publishedID

    def republish(self):
        """Set the timestamp to current and (TODO) update all observers."""
        self.timestamp = time.time()

    def view_getStateToPublish(self, perspective):
        """(internal)"""
        return self.getStateToPublishFor(perspective)

    def getStateToPublishFor(self, perspective):
        """Implement me to special-case your state for a perspective."""
        return self.getStateToPublish()

    def getStateToPublish(self):
        """Implement me to return state to copy as part of the publish phase."""
        raise NotImplementedError('%s.getStateToPublishFor' % self.__class__)

    def getStateToCacheAndObserveFor(self, perspective, observer):
        """Get all necessary metadata to keep a clientside cache."""
        if perspective:
            pname = perspective.perspectiveName
            sname = perspective.getService().serviceName
        else:
            pname = 'None'
            sname = 'None'
        return {'remote': flavors.ViewPoint(perspective, self), 'publishedID': self.publishedID, 'perspective': pname, 'service': sname, 'timestamp': self.timestamp}