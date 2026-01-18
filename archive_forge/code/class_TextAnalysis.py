import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
class TextAnalysis(com.COMObject):
    _interfaces_ = [IDWriteTextAnalysisSource, IDWriteTextAnalysisSink]

    def __init__(self):
        super().__init__()
        self._textstart = 0
        self._textlength = 0
        self._glyphstart = 0
        self._glyphcount = 0
        self._ptrs = []
        self._script = None
        self._bidi = 0

    def GenerateResults(self, analyzer, text, text_length):
        self._text = text
        self._textstart = 0
        self._textlength = text_length
        self._glyphstart = 0
        self._glyphcount = 0
        self._ptrs.clear()
        self._start_run = Run()
        self._start_run.text_length = text_length
        self._current_run = self._start_run
        analyzer.AnalyzeScript(self, 0, text_length, self)

    def SetScriptAnalysis(self, textPosition, textLength, scriptAnalysis):
        self.SetCurrentRun(textPosition)
        self.SplitCurrentRun(textPosition)
        while textLength > 0:
            run, textLength = self.FetchNextRun(textLength)
            run.script.script = scriptAnalysis[0].script
            run.script.shapes = scriptAnalysis[0].shapes
            self._script = run.script
        return 0

    def GetTextBeforePosition(self, textPosition, textString, textLength):
        raise Exception('Currently not implemented.')

    def GetTextAtPosition(self, textPosition, textString, textLength):
        if textPosition >= self._textlength:
            self._no_ptr = c_wchar_p(None)
            textString[0] = self._no_ptr
            textLength[0] = 0
        else:
            ptr = c_wchar_p(self._text[textPosition:])
            self._ptrs.append(ptr)
            textString[0] = ptr
            textLength[0] = self._textlength - textPosition
        return 0

    def GetParagraphReadingDirection(self):
        return 0

    def GetLocaleName(self, textPosition, textLength, localeName):
        self.__local_name = c_wchar_p('')
        localeName[0] = self.__local_name
        textLength[0] = self._textlength - textPosition
        return 0

    def GetNumberSubstitution(self):
        return 0

    def SetCurrentRun(self, textPosition):
        if self._current_run and self._current_run.ContainsTextPosition(textPosition):
            return

    def SplitCurrentRun(self, textPosition):
        if not self._current_run:
            return
        if textPosition <= self._current_run.text_start:
            return
        new_run = copy.copy(self._current_run)
        new_run.next_run = self._current_run.next_run
        self._current_run.next_run = new_run
        splitPoint = textPosition - self._current_run.text_start
        new_run.text_start += splitPoint
        new_run.text_length -= splitPoint
        self._current_run.text_length = splitPoint
        self._current_run = new_run

    def FetchNextRun(self, textLength):
        original_run = self._current_run
        if textLength < self._current_run.text_length:
            self.SplitCurrentRun(self._current_run.text_start + textLength)
        else:
            self._current_run = self._current_run.next_run
        textLength -= original_run.text_length
        return (original_run, textLength)