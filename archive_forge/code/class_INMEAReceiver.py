from zope.interface import Attribute, Interface
class INMEAReceiver(Interface):
    """
    An object that can receive NMEA data.
    """

    def sentenceReceived(sentence):
        """
        Method called when a sentence is received.

        @param sentence: The received NMEA sentence.
        @type L{twisted.positioning.nmea.NMEASentence}
        """