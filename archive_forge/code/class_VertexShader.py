from OpenGL.GL import *  # noqa
from OpenGL.GL import shaders  # noqa
import numpy as np
import re
class VertexShader(Shader):

    def __init__(self, code):
        Shader.__init__(self, GL_VERTEX_SHADER, code)