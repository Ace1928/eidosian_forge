from __future__ import unicode_literals
from PySide2 import QtCore, QtGui, QtWidgets
import classwizard_rc
class ClassWizard(QtWidgets.QWizard):

    def __init__(self, parent=None):
        super(ClassWizard, self).__init__(parent)
        self.addPage(IntroPage())
        self.addPage(ClassInfoPage())
        self.addPage(CodeStylePage())
        self.addPage(OutputFilesPage())
        self.addPage(ConclusionPage())
        self.setPixmap(QtWidgets.QWizard.BannerPixmap, QtGui.QPixmap(':/images/banner.png'))
        self.setPixmap(QtWidgets.QWizard.BackgroundPixmap, QtGui.QPixmap(':/images/background.png'))
        self.setWindowTitle('Class Wizard')

    def accept(self):
        className = self.field('className')
        baseClass = self.field('baseClass')
        macroName = self.field('macroName')
        baseInclude = self.field('baseInclude')
        outputDir = self.field('outputDir')
        header = self.field('header')
        implementation = self.field('implementation')
        block = ''
        if self.field('comment'):
            block += '/*\n'
            block += '    ' + header + '\n'
            block += '*/\n'
            block += '\n'
        if self.field('protect'):
            block += '#ifndef ' + macroName + '\n'
            block += '#define ' + macroName + '\n'
            block += '\n'
        if self.field('includeBase'):
            block += '#include ' + baseInclude + '\n'
            block += '\n'
        block += 'class ' + className
        if baseClass:
            block += ' : public ' + baseClass
        block += '\n'
        block += '{\n'
        if self.field('qobjectMacro'):
            block += '    Q_OBJECT\n'
            block += '\n'
        block += 'public:\n'
        if self.field('qobjectCtor'):
            block += '    ' + className + '(QObject *parent = 0);\n'
        elif self.field('qwidgetCtor'):
            block += '    ' + className + '(QWidget *parent = 0);\n'
        elif self.field('defaultCtor'):
            block += '    ' + className + '();\n'
            if self.field('copyCtor'):
                block += '    ' + className + '(const ' + className + ' &other);\n'
                block += '\n'
                block += '    ' + className + ' &operator=' + '(const ' + className + ' &other);\n'
        block += '};\n'
        if self.field('protect'):
            block += '\n'
            block += '#endif\n'
        headerFile = QtCore.QFile(outputDir + '/' + header)
        if not headerFile.open(QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            QtWidgets.QMessageBox.warning(None, 'Class Wizard', 'Cannot write file %s:\n%s' % (headerFile.fileName(), headerFile.errorString()))
            return
        headerFile.write(QtCore.QByteArray(block.encode('utf-8')))
        block = ''
        if self.field('comment'):
            block += '/*\n'
            block += '    ' + implementation + '\n'
            block += '*/\n'
            block += '\n'
        block += '#include "' + header + '"\n'
        block += '\n'
        if self.field('qobjectCtor'):
            block += className + '::' + className + '(QObject *parent)\n'
            block += '    : ' + baseClass + '(parent)\n'
            block += '{\n'
            block += '}\n'
        elif self.field('qwidgetCtor'):
            block += className + '::' + className + '(QWidget *parent)\n'
            block += '    : ' + baseClass + '(parent)\n'
            block += '{\n'
            block += '}\n'
        elif self.field('defaultCtor'):
            block += className + '::' + className + '()\n'
            block += '{\n'
            block += '    // missing code\n'
            block += '}\n'
            if self.field('copyCtor'):
                block += '\n'
                block += className + '::' + className + '(const ' + className + ' &other)\n'
                block += '{\n'
                block += '    *this = other;\n'
                block += '}\n'
                block += '\n'
                block += className + ' &' + className + '::operator=(const ' + className + ' &other)\n'
                block += '{\n'
                if baseClass:
                    block += '    ' + baseClass + '::operator=(other);\n'
                block += '    // missing code\n'
                block += '    return *this;\n'
                block += '}\n'
        implementationFile = QtCore.QFile(outputDir + '/' + implementation)
        if not implementationFile.open(QtCore.QFile.WriteOnly | QtCore.QFile.Text):
            QtWidgets.QMessageBox.warning(None, 'Class Wizard', 'Cannot write file %s:\n%s' % (implementationFile.fileName(), implementationFile.errorString()))
            return
        implementationFile.write(QtCore.QByteArray(block.encode('utf-8')))
        super(ClassWizard, self).accept()