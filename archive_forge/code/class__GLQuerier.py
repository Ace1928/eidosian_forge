from OpenGL.latebind import LateBind
from OpenGL._bytes import bytes,unicode,as_8_bit
import OpenGL as root
import sys
import logging
class _GLQuerier(ExtensionQuerier):
    prefix = as_8_bit('GL_')
    version_prefix = as_8_bit('GL_VERSION_GL_')
    assumed_version = [1, 1]

    def pullVersion(self):
        """Retrieve 2-int declaration of major/minor GL version

        returns [int(major),int(minor)] or False if not loaded
        """
        from OpenGL import platform
        if not platform.PLATFORM.CurrentContextIsValid():
            return False
        from OpenGL.raw.GL.VERSION.GL_1_1 import glGetString
        from OpenGL.raw.GL.VERSION.GL_1_1 import GL_VERSION
        new = glGetString(GL_VERSION)
        self.version_string = new
        if new:
            return [int(x) for x in new.split(as_8_bit(' '), 1)[0].split(as_8_bit('.'))]
        else:
            return False

    def pullExtensions(self):
        from OpenGL import platform
        if not platform.PLATFORM.CurrentContextIsValid():
            return False
        from OpenGL.raw.GL._types import GLint
        from OpenGL.raw.GL.VERSION.GL_1_1 import glGetString, glGetError
        from OpenGL.raw.GL.VERSION.GL_1_1 import GL_EXTENSIONS
        from OpenGL import error
        try:
            extensions = glGetString(GL_EXTENSIONS)
            if glGetError():
                raise error.GLError()
            if extensions:
                extensions = extensions.split()
            else:
                return False
        except (AttributeError, error.GLError):
            from OpenGL.raw.GL.VERSION.GL_3_0 import GL_NUM_EXTENSIONS, glGetStringi
            from OpenGL.raw.GL.VERSION.GL_1_1 import glGetIntegerv
            count = GLint()
            glGetIntegerv(GL_NUM_EXTENSIONS, count)
            extensions = []
            for i in range(count.value):
                extension = glGetStringi(GL_EXTENSIONS, i)
                extensions.append(extension)
        version = self.getVersion()
        if not version:
            return version
        check = tuple(version[:2])
        for v, v_exts in VERSION_EXTENSIONS:
            if v <= check:
                for v_ext in v_exts:
                    if v_ext not in extensions:
                        extensions.append(as_8_bit(v_ext))
            else:
                break
        return extensions