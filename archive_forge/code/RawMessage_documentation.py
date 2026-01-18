from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer

  This is a special subclass of ProtocolMessage that doesn't interpret its data
  in any way. Instead, it just stores it in a string.

  See rawmessage.h for more details.
  