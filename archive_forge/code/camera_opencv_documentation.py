from __future__ import division
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.camera import CameraBase

            On OSX, not only are the import names different,
            but the API also differs.
            There is no module called 'highgui' but the names are
            directly available in the 'cv' module.
            Some of them even have a different names.

            Therefore we use this proxy object.
            