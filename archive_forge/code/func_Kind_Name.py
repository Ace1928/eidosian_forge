from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
import abc
import array
def Kind_Name(cls, x):
    return cls._Kind_NAMES.get(x, '')