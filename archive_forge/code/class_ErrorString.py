import sys, codecs
class ErrorString(SafeString):
    """
    Safely report exception type and message.
    """

    def __str__(self):
        return '%s: %s' % (self.data.__class__.__name__, super(ErrorString, self).__str__())

    def __unicode__(self):
        return '%s: %s' % (self.data.__class__.__name__, super(ErrorString, self).__unicode__())