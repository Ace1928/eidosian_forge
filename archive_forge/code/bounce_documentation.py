import email.utils
import os
import time
from io import SEEK_END, SEEK_SET, StringIO
from twisted.mail import smtp

    Generate a bounce message for an undeliverable email message.

    @type message: a file-like object
    @param message: The undeliverable message.

    @type failedFrom: L{bytes} or L{unicode}
    @param failedFrom: The originator of the undeliverable message.

    @type failedTo: L{bytes} or L{unicode}
    @param failedTo: The destination of the undeliverable message.

    @type transcript: L{bytes} or L{unicode}
    @param transcript: An error message to include in the bounce message.

    @type encoding: L{str} or L{unicode}
    @param encoding: Encoding to use, default: utf-8

    @rtype: 3-L{tuple} of (E{1}) L{bytes}, (E{2}) L{bytes}, (E{3}) L{bytes}
    @return: The originator, the destination and the contents of the bounce
        message.  The destination of the bounce message is the originator of
        the undeliverable message.
    