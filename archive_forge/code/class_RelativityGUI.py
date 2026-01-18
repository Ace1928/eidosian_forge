import collections
import os
import sys
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree import types as pTypes
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
class RelativityGUI(QtWidgets.QWidget):

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.animations = []
        self.animTimer = QtCore.QTimer()
        self.animTimer.timeout.connect(self.stepAnimation)
        self.animTime = 0
        self.animDt = 0.016
        self.lastAnimTime = 0
        self.setupGUI()
        self.objectGroup = ObjectGroupParam()
        self.params = Parameter.create(name='params', type='group', children=[dict(name='Load Preset..', type='list', limits=[]), dict(name='Duration', type='float', value=10.0, step=0.1, limits=[0.1, None]), dict(name='Reference Frame', type='list', limits=[]), dict(name='Animate', type='bool', value=True), dict(name='Animation Speed', type='float', value=1.0, dec=True, step=0.1, limits=[0.0001, None]), dict(name='Recalculate Worldlines', type='action'), dict(name='Save', type='action'), dict(name='Load', type='action'), self.objectGroup])
        self.tree.setParameters(self.params, showTop=False)
        self.params.param('Recalculate Worldlines').sigActivated.connect(self.recalculate)
        self.params.param('Save').sigActivated.connect(self.save)
        self.params.param('Load').sigActivated.connect(self.load)
        self.params.param('Load Preset..').sigValueChanged.connect(self.loadPreset)
        self.params.sigTreeStateChanged.connect(self.treeChanged)
        presetDir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'presets')
        if os.path.exists(presetDir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(presetDir)]
            self.params.param('Load Preset..').setLimits([''] + presets)

    def setupGUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.splitter = QtWidgets.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.layout.addWidget(self.splitter)
        self.tree = ParameterTree(showHeader=False)
        self.splitter.addWidget(self.tree)
        self.splitter2 = QtWidgets.QSplitter()
        self.splitter2.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter.addWidget(self.splitter2)
        self.worldlinePlots = pg.GraphicsLayoutWidget()
        self.splitter2.addWidget(self.worldlinePlots)
        self.animationPlots = pg.GraphicsLayoutWidget()
        self.splitter2.addWidget(self.animationPlots)
        self.splitter2.setSizes([int(self.height() * 0.8), int(self.height() * 0.2)])
        self.inertWorldlinePlot = self.worldlinePlots.addPlot()
        self.refWorldlinePlot = self.worldlinePlots.addPlot()
        self.inertAnimationPlot = self.animationPlots.addPlot()
        self.inertAnimationPlot.setAspectLocked(1)
        self.refAnimationPlot = self.animationPlots.addPlot()
        self.refAnimationPlot.setAspectLocked(1)
        self.inertAnimationPlot.setXLink(self.inertWorldlinePlot)
        self.refAnimationPlot.setXLink(self.refWorldlinePlot)

    def recalculate(self):
        clocks1 = collections.OrderedDict()
        clocks2 = collections.OrderedDict()
        for cl in self.params.param('Objects'):
            clocks1.update(cl.buildClocks())
            clocks2.update(cl.buildClocks())
        dt = self.animDt * self.params['Animation Speed']
        sim1 = Simulation(clocks1, ref=None, duration=self.params['Duration'], dt=dt)
        sim1.run()
        sim1.plot(self.inertWorldlinePlot)
        self.inertWorldlinePlot.autoRange(padding=0.1)
        ref = self.params['Reference Frame']
        dur = clocks1[ref].refData['pt'][-1]
        sim2 = Simulation(clocks2, ref=clocks2[ref], duration=dur, dt=dt)
        sim2.run()
        sim2.plot(self.refWorldlinePlot)
        self.refWorldlinePlot.autoRange(padding=0.1)
        self.refAnimationPlot.clear()
        self.inertAnimationPlot.clear()
        self.animTime = 0
        self.animations = [Animation(sim1), Animation(sim2)]
        self.inertAnimationPlot.addItem(self.animations[0])
        self.refAnimationPlot.addItem(self.animations[1])
        self.inertWorldlinePlot.addItem(self.animations[0].items[ref].spaceline())
        self.refWorldlinePlot.addItem(self.animations[1].items[ref].spaceline())

    def setAnimation(self, a):
        if a:
            self.lastAnimTime = perf_counter()
            self.animTimer.start(int(self.animDt * 1000))
        else:
            self.animTimer.stop()

    def stepAnimation(self):
        now = perf_counter()
        dt = (now - self.lastAnimTime) * self.params['Animation Speed']
        self.lastAnimTime = now
        self.animTime += dt
        if self.animTime > self.params['Duration']:
            self.animTime = 0
            for a in self.animations:
                a.restart()
        for a in self.animations:
            a.stepTo(self.animTime)

    def treeChanged(self, *args):
        clocks = []
        for c in self.params.param('Objects'):
            clocks.extend(c.clockNames())
        self.params.param('Reference Frame').setLimits(clocks)
        self.setAnimation(self.params['Animate'])

    def save(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save State..', 'untitled.cfg', 'Config Files (*.cfg)')
        if not filename:
            return
        state = self.params.saveState()
        configfile.writeConfigFile(state, filename)

    def load(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Save State..', '', 'Config Files (*.cfg)')
        if not filename:
            return
        state = configfile.readConfigFile(filename)
        self.loadState(state)

    def loadPreset(self, param, preset):
        if preset == '':
            return
        path = os.path.abspath(os.path.dirname(__file__))
        fn = os.path.join(path, 'presets', preset + '.cfg')
        state = configfile.readConfigFile(fn)
        self.loadState(state)

    def loadState(self, state):
        if 'Load Preset..' in state['children']:
            del state['children']['Load Preset..']['limits']
            del state['children']['Load Preset..']['value']
        self.params.param('Objects').clearChildren()
        self.params.restoreState(state, removeChildren=False)
        self.recalculate()