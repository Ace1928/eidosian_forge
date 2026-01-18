import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class ArduinoLexer(CppLexer):
    """
    For `Arduino(tm) <https://arduino.cc/>`_ source.

    This is an extension of the CppLexer, as the Arduino® Language is a superset
    of C++

    .. versionadded:: 2.1
    """
    name = 'Arduino'
    aliases = ['arduino']
    filenames = ['*.ino']
    mimetypes = ['text/x-arduino']
    structure = set(('setup', 'loop'))
    operators = set(('not', 'or', 'and', 'xor'))
    variables = set(('DIGITAL_MESSAGE', 'FIRMATA_STRING', 'ANALOG_MESSAGE', 'REPORT_DIGITAL', 'REPORT_ANALOG', 'INPUT_PULLUP', 'SET_PIN_MODE', 'INTERNAL2V56', 'SYSTEM_RESET', 'LED_BUILTIN', 'INTERNAL1V1', 'SYSEX_START', 'INTERNAL', 'EXTERNAL', 'HIGH', 'LOW', 'INPUT', 'OUTPUT', 'INPUT_PULLUP', 'LED_BUILTIN', 'true', 'false', 'void', 'boolean', 'char', 'unsigned char', 'byte', 'int', 'unsigned int', 'word', 'long', 'unsigned long', 'short', 'float', 'double', 'string', 'String', 'array', 'static', 'volatile', 'const', 'boolean', 'byte', 'word', 'string', 'String', 'array', 'int', 'float', 'private', 'char', 'virtual', 'operator', 'sizeof', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'dynamic_cast', 'typedef', 'const_cast', 'const', 'struct', 'static_cast', 'union', 'unsigned', 'long', 'volatile', 'static', 'protected', 'bool', 'public', 'friend', 'auto', 'void', 'enum', 'extern', 'class', 'short', 'reinterpret_cast', 'double', 'register', 'explicit', 'signed', 'inline', 'delete', '_Bool', 'complex', '_Complex', '_Imaginary', 'atomic_bool', 'atomic_char', 'atomic_schar', 'atomic_uchar', 'atomic_short', 'atomic_ushort', 'atomic_int', 'atomic_uint', 'atomic_long', 'atomic_ulong', 'atomic_llong', 'atomic_ullong', 'PROGMEM'))
    functions = set(('KeyboardController', 'MouseController', 'SoftwareSerial', 'EthernetServer', 'EthernetClient', 'LiquidCrystal', 'RobotControl', 'GSMVoiceCall', 'EthernetUDP', 'EsploraTFT', 'HttpClient', 'RobotMotor', 'WiFiClient', 'GSMScanner', 'FileSystem', 'Scheduler', 'GSMServer', 'YunClient', 'YunServer', 'IPAddress', 'GSMClient', 'GSMModem', 'Keyboard', 'Ethernet', 'Console', 'GSMBand', 'Esplora', 'Stepper', 'Process', 'WiFiUDP', 'GSM_SMS', 'Mailbox', 'USBHost', 'Firmata', 'PImage', 'Client', 'Server', 'GSMPIN', 'FileIO', 'Bridge', 'Serial', 'EEPROM', 'Stream', 'Mouse', 'Audio', 'Servo', 'File', 'Task', 'GPRS', 'WiFi', 'Wire', 'TFT', 'GSM', 'SPI', 'SD', 'runShellCommandAsynchronously', 'analogWriteResolution', 'retrieveCallingNumber', 'printFirmwareVersion', 'analogReadResolution', 'sendDigitalPortPair', 'noListenOnLocalhost', 'readJoystickButton', 'setFirmwareVersion', 'readJoystickSwitch', 'scrollDisplayRight', 'getVoiceCallStatus', 'scrollDisplayLeft', 'writeMicroseconds', 'delayMicroseconds', 'beginTransmission', 'getSignalStrength', 'runAsynchronously', 'getAsynchronously', 'listenOnLocalhost', 'getCurrentCarrier', 'readAccelerometer', 'messageAvailable', 'sendDigitalPorts', 'lineFollowConfig', 'countryNameWrite', 'runShellCommand', 'readStringUntil', 'rewindDirectory', 'readTemperature', 'setClockDivider', 'readLightSensor', 'endTransmission', 'analogReference', 'detachInterrupt', 'countryNameRead', 'attachInterrupt', 'encryptionType', 'readBytesUntil', 'robotNameWrite', 'readMicrophone', 'robotNameRead', 'cityNameWrite', 'userNameWrite', 'readJoystickY', 'readJoystickX', 'mouseReleased', 'openNextFile', 'scanNetworks', 'noInterrupts', 'digitalWrite', 'beginSpeaker', 'mousePressed', 'isActionDone', 'mouseDragged', 'displayLogos', 'noAutoscroll', 'addParameter', 'remoteNumber', 'getModifiers', 'keyboardRead', 'userNameRead', 'waitContinue', 'processInput', 'parseCommand', 'printVersion', 'readNetworks', 'writeMessage', 'blinkVersion', 'cityNameRead', 'readMessage', 'setDataMode', 'parsePacket', 'isListening', 'setBitOrder', 'beginPacket', 'isDirectory', 'motorsWrite', 'drawCompass', 'digitalRead', 'clearScreen', 'serialEvent', 'rightToLeft', 'setTextSize', 'leftToRight', 'requestFrom', 'keyReleased', 'compassRead', 'analogWrite', 'interrupts', 'WiFiServer', 'disconnect', 'playMelody', 'parseFloat', 'autoscroll', 'getPINUsed', 'setPINUsed', 'setTimeout', 'sendAnalog', 'readSlider', 'analogRead', 'beginWrite', 'createChar', 'motorsStop', 'keyPressed', 'tempoWrite', 'readButton', 'subnetMask', 'debugPrint', 'macAddress', 'writeGreen', 'randomSeed', 'attachGPRS', 'readString', 'sendString', 'remotePort', 'releaseAll', 'mouseMoved', 'background', 'getXChange', 'getYChange', 'answerCall', 'getResult', 'voiceCall', 'endPacket', 'constrain', 'getSocket', 'writeJSON', 'getButton', 'available', 'connected', 'findUntil', 'readBytes', 'exitValue', 'readGreen', 'writeBlue', 'startLoop', 'IPAddress', 'isPressed', 'sendSysex', 'pauseMode', 'gatewayIP', 'setCursor', 'getOemKey', 'tuneWrite', 'noDisplay', 'loadImage', 'switchPIN', 'onRequest', 'onReceive', 'changePIN', 'playFile', 'noBuffer', 'parseInt', 'overflow', 'checkPIN', 'knobRead', 'beginTFT', 'bitClear', 'updateIR', 'bitWrite', 'position', 'writeRGB', 'highByte', 'writeRed', 'setSpeed', 'readBlue', 'noStroke', 'remoteIP', 'transfer', 'shutdown', 'hangCall', 'beginSMS', 'endWrite', 'attached', 'maintain', 'noCursor', 'checkReg', 'checkPUK', 'shiftOut', 'isValid', 'shiftIn', 'pulseIn', 'connect', 'println', 'localIP', 'pinMode', 'getIMEI', 'display', 'noBlink', 'process', 'getBand', 'running', 'beginSD', 'drawBMP', 'lowByte', 'setBand', 'release', 'bitRead', 'prepare', 'pointTo', 'readRed', 'setMode', 'noFill', 'remove', 'listen', 'stroke', 'detach', 'attach', 'noTone', 'exists', 'buffer', 'height', 'bitSet', 'circle', 'config', 'cursor', 'random', 'IRread', 'setDNS', 'endSMS', 'getKey', 'micros', 'millis', 'begin', 'print', 'write', 'ready', 'flush', 'width', 'isPIN', 'blink', 'clear', 'press', 'mkdir', 'rmdir', 'close', 'point', 'yield', 'image', 'BSSID', 'click', 'delay', 'read', 'text', 'move', 'peek', 'beep', 'rect', 'line', 'open', 'seek', 'fill', 'size', 'turn', 'stop', 'home', 'find', 'step', 'tone', 'sqrt', 'RSSI', 'SSID', 'end', 'bit', 'tan', 'cos', 'sin', 'pow', 'map', 'abs', 'max', 'min', 'get', 'run', 'put', 'isAlphaNumeric', 'isAlpha', 'isAscii', 'isWhitespace', 'isControl', 'isDigit', 'isGraph', 'isLowerCase', 'isPrintable', 'isPunct', 'isSpace', 'isUpperCase', 'isHexadecimalDigit'))
    suppress_highlight = set(('namespace', 'template', 'mutable', 'using', 'asm', 'typeid', 'typename', 'this', 'alignof', 'constexpr', 'decltype', 'noexcept', 'static_assert', 'thread_local', 'restrict'))

    def get_tokens_unprocessed(self, text):
        for index, token, value in CppLexer.get_tokens_unprocessed(self, text):
            if value in self.structure:
                yield (index, Name.Builtin, value)
            elif value in self.operators:
                yield (index, Operator, value)
            elif value in self.variables:
                yield (index, Keyword.Reserved, value)
            elif value in self.suppress_highlight:
                yield (index, Name, value)
            elif value in self.functions:
                yield (index, Name.Function, value)
            else:
                yield (index, token, value)