from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.core import log
def GetTypeIds(self):
    """Return a list of all type ids in the set.

    Returns:
      [ cls1.MESSAGE_TYPE_ID, ... ] for each cls in the set.  The returned
      list does not contain duplicates.
    """
    return self.items.keys()