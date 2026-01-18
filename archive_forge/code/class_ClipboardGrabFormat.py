import threading
import numpy as np
from ..core import Format
class ClipboardGrabFormat(BaseGrabFormat):
    """The ClipboardGrabFormat provided a means to grab image data from
    the clipboard, using the uri "<clipboard>"

    This functionality is provided via Pillow. Note that "<clipboard>" is
    only supported on Windows.

    Parameters for reading
    ----------------------
    No parameters.
    """

    def _can_read(self, request):
        if request.filename != '<clipboard>':
            return False
        return bool(self._init_pillow())

    def _get_data(self, index):
        ImageGrab = self._init_pillow()
        assert ImageGrab
        pil_im = ImageGrab.grabclipboard()
        if pil_im is None:
            raise RuntimeError('There seems to be no image data on the clipboard now.')
        im = np.asarray(pil_im)
        return (im, {})