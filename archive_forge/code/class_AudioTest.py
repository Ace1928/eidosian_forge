from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
class AudioTest(QMainWindow):
    PUSH_MODE_LABEL = 'Enable push mode'
    PULL_MODE_LABEL = 'Enable pull mode'
    SUSPEND_LABEL = 'Suspend playback'
    RESUME_LABEL = 'Resume playback'
    DurationSeconds = 1
    ToneSampleRateHz = 600
    DataSampleRateHz = 44100

    def __init__(self):
        super(AudioTest, self).__init__()
        self.m_device = QAudioDeviceInfo.defaultOutputDevice()
        self.m_output = None
        self.initializeWindow()
        self.initializeAudio()

    def initializeWindow(self):
        layout = QVBoxLayout()
        self.m_deviceBox = QComboBox()
        self.m_deviceBox.activated[int].connect(self.deviceChanged)
        for deviceInfo in QAudioDeviceInfo.availableDevices(QAudio.AudioOutput):
            self.m_deviceBox.addItem(deviceInfo.deviceName(), deviceInfo)
        layout.addWidget(self.m_deviceBox)
        self.m_modeButton = QPushButton()
        self.m_modeButton.clicked.connect(self.toggleMode)
        self.m_modeButton.setText(self.PUSH_MODE_LABEL)
        layout.addWidget(self.m_modeButton)
        self.m_suspendResumeButton = QPushButton(clicked=self.toggleSuspendResume)
        self.m_suspendResumeButton.setText(self.SUSPEND_LABEL)
        layout.addWidget(self.m_suspendResumeButton)
        volumeBox = QHBoxLayout()
        volumeLabel = QLabel('Volume:')
        self.m_volumeSlider = QSlider(Qt.Horizontal, minimum=0, maximum=100, singleStep=10)
        self.m_volumeSlider.valueChanged.connect(self.volumeChanged)
        volumeBox.addWidget(volumeLabel)
        volumeBox.addWidget(self.m_volumeSlider)
        layout.addLayout(volumeBox)
        window = QWidget()
        window.setLayout(layout)
        self.setCentralWidget(window)

    def initializeAudio(self):
        self.m_pullTimer = QTimer(self)
        self.m_pullTimer.timeout.connect(self.pullTimerExpired)
        self.m_pullMode = True
        self.m_format = QAudioFormat()
        self.m_format.setSampleRate(self.DataSampleRateHz)
        self.m_format.setChannelCount(1)
        self.m_format.setSampleSize(16)
        self.m_format.setCodec('audio/pcm')
        self.m_format.setByteOrder(QAudioFormat.LittleEndian)
        self.m_format.setSampleType(QAudioFormat.SignedInt)
        info = QAudioDeviceInfo(QAudioDeviceInfo.defaultOutputDevice())
        if not info.isFormatSupported(self.m_format):
            qWarning('Default format not supported - trying to use nearest')
            self.m_format = info.nearestFormat(self.m_format)
        self.m_generator = Generator(self.m_format, self.DurationSeconds * 1000000, self.ToneSampleRateHz, self)
        self.createAudioOutput()

    def createAudioOutput(self):
        self.m_audioOutput = QAudioOutput(self.m_device, self.m_format)
        self.m_audioOutput.notify.connect(self.notified)
        self.m_audioOutput.stateChanged.connect(self.handleStateChanged)
        self.m_generator.start()
        self.m_audioOutput.start(self.m_generator)
        self.m_volumeSlider.setValue(self.m_audioOutput.volume() * 100)

    def deviceChanged(self, index):
        self.m_pullTimer.stop()
        self.m_generator.stop()
        self.m_audioOutput.stop()
        self.m_device = self.m_deviceBox.itemData(index)
        self.createAudioOutput()

    def volumeChanged(self, value):
        if self.m_audioOutput is not None:
            self.m_audioOutput.setVolume(value / 100.0)

    def notified(self):
        qWarning('bytesFree = %d, elapsedUSecs = %d, processedUSecs = %d' % (self.m_audioOutput.bytesFree(), self.m_audioOutput.elapsedUSecs(), self.m_audioOutput.processedUSecs()))

    def pullTimerExpired(self):
        if self.m_audioOutput is not None and self.m_audioOutput.state() != QAudio.StoppedState:
            chunks = self.m_audioOutput.bytesFree() // self.m_audioOutput.periodSize()
            for _ in range(chunks):
                data = self.m_generator.read(self.m_audioOutput.periodSize())
                if data is None or len(data) != self.m_audioOutput.periodSize():
                    break
                self.m_output.write(data)

    def toggleMode(self):
        self.m_pullTimer.stop()
        self.m_audioOutput.stop()
        if self.m_pullMode:
            self.m_modeButton.setText(self.PULL_MODE_LABEL)
            self.m_output = self.m_audioOutput.start()
            self.m_pullMode = False
            self.m_pullTimer.start(20)
        else:
            self.m_modeButton.setText(self.PUSH_MODE_LABEL)
            self.m_pullMode = True
            self.m_audioOutput.start(self.m_generator)
        self.m_suspendResumeButton.setText(self.SUSPEND_LABEL)

    def toggleSuspendResume(self):
        if self.m_audioOutput.state() == QAudio.SuspendedState:
            qWarning('status: Suspended, resume()')
            self.m_audioOutput.resume()
            self.m_suspendResumeButton.setText(self.SUSPEND_LABEL)
        elif self.m_audioOutput.state() == QAudio.ActiveState:
            qWarning('status: Active, suspend()')
            self.m_audioOutput.suspend()
            self.m_suspendResumeButton.setText(self.RESUME_LABEL)
        elif self.m_audioOutput.state() == QAudio.StoppedState:
            qWarning('status: Stopped, resume()')
            self.m_audioOutput.resume()
            self.m_suspendResumeButton.setText(self.SUSPEND_LABEL)
        elif self.m_audioOutput.state() == QAudio.IdleState:
            qWarning('status: IdleState')
    stateMap = {QAudio.ActiveState: 'ActiveState', QAudio.SuspendedState: 'SuspendedState', QAudio.StoppedState: 'StoppedState', QAudio.IdleState: 'IdleState'}

    def handleStateChanged(self, state):
        qWarning('state = ' + self.stateMap.get(state, 'Unknown'))