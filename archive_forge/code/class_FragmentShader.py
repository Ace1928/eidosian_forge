from OpenGL.GL import *  # noqa
from OpenGL.GL import shaders  # noqa
import numpy as np
import re
class FragmentShader(Shader):

    def __init__(self, code):
        Shader.__init__(self, GL_FRAGMENT_SHADER, code)