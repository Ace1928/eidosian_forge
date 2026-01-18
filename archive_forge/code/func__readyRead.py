import os
import sys
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import QPointF, QRect, QSize
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import QApplication, QMainWindow, QMessageBox
def _readyRead(self):
    data = self.ioDevice.readAll()
    availableSamples = data.size() // resolution
    start = 0
    if availableSamples < sampleCount:
        start = sampleCount - availableSamples
        for s in range(start):
            self.buffer[s].setY(self.buffer[s + availableSamples].y())
    dataIndex = 0
    for s in range(start, sampleCount):
        value = (ord(data[dataIndex]) - 128) / 128
        self.buffer[s].setY(value)
        dataIndex = dataIndex + resolution
    self.series.replace(self.buffer)