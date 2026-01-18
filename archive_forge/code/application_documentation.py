from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
Create an application/* type MIME document.

        _data contains the bytes for the raw application data.

        _subtype is the MIME content type subtype, defaulting to
        'octet-stream'.

        _encoder is a function which will perform the actual encoding for
        transport of the application data, defaulting to base64 encoding.

        Any additional keyword arguments are passed to the base class
        constructor, which turns them into parameters on the Content-Type
        header.
        