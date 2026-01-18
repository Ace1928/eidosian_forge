from suds import UnicodeMixin
import sys
def __set_URL(self, url):
    """
        URL is stored as a str internally and must not contain ASCII chars.

        Raised exception in case of detected non-ASCII URL characters may be
        either UnicodeEncodeError or UnicodeDecodeError, depending on the used
        Python version's str type and the exact value passed as URL input data.

        """
    if isinstance(url, str):
        url.encode('ascii')
        self.url = url
    elif sys.version_info < (3, 0):
        self.url = url.encode('ascii')
    else:
        self.url = url.decode('ascii')