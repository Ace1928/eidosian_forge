import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
class MouseDragHandler(object):
    """Implements default mouse drag behavior for ROI (not for ROI handles).
    """

    def __init__(self, roi):
        self.roi = roi
        self.dragMode = None
        self.startState = None
        self.snapModifier = QtCore.Qt.KeyboardModifier.ControlModifier
        self.translateModifier = QtCore.Qt.KeyboardModifier.NoModifier
        self.rotateModifier = QtCore.Qt.KeyboardModifier.AltModifier
        self.scaleModifier = QtCore.Qt.KeyboardModifier.ShiftModifier
        self.rotateSpeed = 0.5
        self.scaleSpeed = 1.01

    def mouseDragEvent(self, ev):
        roi = self.roi
        if ev.isStart():
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                roi.setSelected(True)
                mods = ev.modifiers()
                try:
                    mods &= ~self.snapModifier
                except ValueError:
                    if mods & self.snapModifier:
                        mods ^= self.snapModifier
                if roi.translatable and mods == self.translateModifier:
                    self.dragMode = 'translate'
                elif roi.rotatable and mods == self.rotateModifier:
                    self.dragMode = 'rotate'
                elif roi.resizable and mods == self.scaleModifier:
                    self.dragMode = 'scale'
                else:
                    self.dragMode = None
                if self.dragMode is not None:
                    roi._moveStarted()
                    self.startPos = roi.mapToParent(ev.buttonDownPos())
                    self.startState = roi.saveState()
                    self.cursorOffset = roi.pos() - self.startPos
                    ev.accept()
                else:
                    ev.ignore()
            else:
                self.dragMode = None
                ev.ignore()
        if ev.isFinish() and self.dragMode is not None:
            roi._moveFinished()
            return
        if not roi.isMoving or self.dragMode is None:
            return
        snap = True if ev.modifiers() & self.snapModifier else None
        pos = roi.mapToParent(ev.pos())
        if self.dragMode == 'translate':
            newPos = pos + self.cursorOffset
            roi.translate(newPos - roi.pos(), snap=snap, finish=False)
        elif self.dragMode == 'rotate':
            diff = self.rotateSpeed * (ev.scenePos() - ev.buttonDownScenePos()).x()
            angle = self.startState['angle'] - diff
            roi.setAngle(angle, centerLocal=ev.buttonDownPos(), snap=snap, finish=False)
        elif self.dragMode == 'scale':
            diff = self.scaleSpeed ** (-(ev.scenePos() - ev.buttonDownScenePos()).y())
            roi.setSize(Point(self.startState['size']) * diff, centerLocal=ev.buttonDownPos(), snap=snap, finish=False)