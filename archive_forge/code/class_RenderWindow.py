from argparse import ArgumentParser, RawTextHelpFormatter
import numpy
import sys
from textwrap import dedent
from PySide2.QtCore import QCoreApplication, QLibraryInfo, QSize, QTimer, Qt
from PySide2.QtGui import (QMatrix4x4, QOpenGLBuffer, QOpenGLContext, QOpenGLShader,
from PySide2.QtWidgets import (QApplication, QHBoxLayout, QMessageBox, QPlainTextEdit,
from PySide2.support import VoidPtr
class RenderWindow(QWindow):

    def __init__(self, format):
        super(RenderWindow, self).__init__()
        self.setSurfaceType(QWindow.OpenGLSurface)
        self.setFormat(format)
        self.context = QOpenGLContext(self)
        self.context.setFormat(self.requestedFormat())
        if not self.context.create():
            raise Exception('Unable to create GL context')
        self.program = None
        self.timer = None
        self.angle = 0

    def initGl(self):
        self.program = QOpenGLShaderProgram(self)
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer()
        format = self.context.format()
        useNewStyleShader = format.profile() == QSurfaceFormat.CoreProfile
        if format.renderableType() == QSurfaceFormat.OpenGL and format.majorVersion() == 3 and (format.minorVersion() <= 1):
            useNewStyleShader = not format.testOption(QSurfaceFormat.DeprecatedFunctions)
        vertexShader = vertexShaderSource if useNewStyleShader else vertexShaderSource110
        fragmentShader = fragmentShaderSource if useNewStyleShader else fragmentShaderSource110
        if not self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertexShader):
            raise Exception('Vertex shader could not be added: {} ({})'.format(self.program.log(), vertexShader))
        if not self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragmentShader):
            raise Exception('Fragment shader could not be added: {} ({})'.format(self.program.log(), fragmentShader))
        if not self.program.link():
            raise Exception('Could not link shaders: {}'.format(self.program.log()))
        self.posAttr = self.program.attributeLocation('posAttr')
        self.colAttr = self.program.attributeLocation('colAttr')
        self.matrixUniform = self.program.uniformLocation('matrix')
        self.vbo.create()
        self.vbo.bind()
        self.verticesData = vertices.tobytes()
        self.colorsData = colors.tobytes()
        verticesSize = 4 * vertices.size
        colorsSize = 4 * colors.size
        self.vbo.allocate(VoidPtr(self.verticesData), verticesSize + colorsSize)
        self.vbo.write(verticesSize, VoidPtr(self.colorsData), colorsSize)
        self.vbo.release()
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)
        if self.vao.isCreated():
            self.setupVertexAttribs()

    def setupVertexAttribs(self):
        self.vbo.bind()
        self.program.setAttributeBuffer(self.posAttr, GL.GL_FLOAT, 0, 2)
        self.program.setAttributeBuffer(self.colAttr, GL.GL_FLOAT, 4 * vertices.size, 3)
        self.program.enableAttributeArray(self.posAttr)
        self.program.enableAttributeArray(self.colAttr)
        self.vbo.release()

    def exposeEvent(self, event):
        if self.isExposed():
            self.render()
            if self.timer is None:
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.slotTimer)
            if not self.timer.isActive():
                self.timer.start(10)
        elif self.timer and self.timer.isActive():
            self.timer.stop()

    def render(self):
        if not self.context.makeCurrent(self):
            raise Exception('makeCurrent() failed')
        functions = self.context.functions()
        if self.program is None:
            functions.glEnable(GL.GL_DEPTH_TEST)
            functions.glClearColor(0, 0, 0, 1)
            self.initGl()
        retinaScale = self.devicePixelRatio()
        functions.glViewport(0, 0, self.width() * retinaScale, self.height() * retinaScale)
        functions.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.program.bind()
        matrix = QMatrix4x4()
        matrix.perspective(60, 4 / 3, 0.1, 100)
        matrix.translate(0, 0, -2)
        matrix.rotate(self.angle, 0, 1, 0)
        self.program.setUniformValue(self.matrixUniform, matrix)
        if self.vao.isCreated():
            self.vao.bind()
        else:
            self.setupVertexAttribs()
        functions.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        self.vao.release()
        self.program.release()
        self.context.swapBuffers(self)
        self.context.doneCurrent()

    def slotTimer(self):
        self.render()
        self.angle += 1

    def glInfo(self):
        if not self.context.makeCurrent(self):
            raise Exception('makeCurrent() failed')
        functions = self.context.functions()
        text = 'Vendor: {}\nRenderer: {}\nVersion: {}\nShading language: {}\n\nContext Format: {}\n\nSurface Format: {}'.format(functions.glGetString(GL.GL_VENDOR), functions.glGetString(GL.GL_RENDERER), functions.glGetString(GL.GL_VERSION), functions.glGetString(GL.GL_SHADING_LANGUAGE_VERSION), print_surface_format(self.context.format()), print_surface_format(self.format()))
        self.context.doneCurrent()
        return text