from math import ceil
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.camera import CameraBase
from picamera import PiCamera
import numpy
Round buffer size up to 32x16 blocks.

        See https://picamera.readthedocs.io/en/release-1.13/recipes2.html#capturing-to-a-numpy-array
        