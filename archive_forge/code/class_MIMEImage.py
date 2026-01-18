from email import encoders
from email.mime.nonmultipart import MIMENonMultipart
class MIMEImage(MIMENonMultipart):
    """Class for generating image/* type MIME documents."""

    def __init__(self, _imagedata, _subtype=None, _encoder=encoders.encode_base64, *, policy=None, **_params):
        """Create an image/* type MIME document.

        _imagedata contains the bytes for the raw image data.  If the data
        type can be detected (jpeg, png, gif, tiff, rgb, pbm, pgm, ppm,
        rast, xbm, bmp, webp, and exr attempted), then the subtype will be
        automatically included in the Content-Type header. Otherwise, you can
        specify the specific image subtype via the _subtype parameter.

        _encoder is a function which will perform the actual encoding for
        transport of the image data.  It takes one argument, which is this
        Image instance.  It should use get_payload() and set_payload() to
        change the payload to the encoded form.  It should also add any
        Content-Transfer-Encoding or other headers to the message as
        necessary.  The default encoding is Base64.

        Any additional keyword arguments are passed to the base class
        constructor, which turns them into parameters on the Content-Type
        header.
        """
        _subtype = _what(_imagedata) if _subtype is None else _subtype
        if _subtype is None:
            raise TypeError('Could not guess image MIME subtype')
        MIMENonMultipart.__init__(self, 'image', _subtype, policy=policy, **_params)
        self.set_payload(_imagedata)
        _encoder(self)