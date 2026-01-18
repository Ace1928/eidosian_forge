from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def Status_Name(cls, x):
    return cls._Status_NAMES.get(x, '')