import os
import tempfile
import time
import re
def CloseChemdraw():
    """ shuts down chemdraw

  """
    global cdApp
    try:
        cdApp.Quit()
    except Exception:
        pass
    Exit()