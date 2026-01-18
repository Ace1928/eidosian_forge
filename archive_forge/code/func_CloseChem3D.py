import os
import tempfile
import time
import re
def CloseChem3D():
    """ shuts down Chem3D """
    global c3dApp
    c3dApp.Quit()
    c3dApp = None