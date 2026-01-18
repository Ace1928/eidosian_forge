import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
class MouseInfoWindow:

    def _updateMouseInfoTextFields(self):
        x, y = position()
        self.xyTextboxSV.set('%s,%s' % (x - self.xOrigin, y - self.yOrigin))
        width, height = size()
        if not _PILLOW_INSTALLED:
            self.rgbSV.set('NA_Pillow_unsupported')
        elif sys.platform == 'darwin':
            self.rgbSV.set('NA_on_macOS')
        elif not (0 <= x < width and 0 <= y < height):
            self.rgbSV.set('NA_on_multimonitor_setups')
        else:
            r, g, b = getPixel(x, y)
            self.rgbSV.set('%s,%s,%s' % (r, g, b))
        if not _PILLOW_INSTALLED:
            self.rgbHexSV.set('NA_Pillow_unsupported')
        elif sys.platform == 'darwin':
            self.rgbHexSV.set('NA_on_macOS')
        elif not (0 <= x < width and 0 <= y < height):
            self.rgbHexSV.set('NA_on_multimonitor_setups')
        else:
            rHex = hex(r)[2:].upper().rjust(2, '0')
            gHex = hex(g)[2:].upper().rjust(2, '0')
            bHex = hex(b)[2:].upper().rjust(2, '0')
            hexColor = '#%s%s%s' % (rHex, gHex, bHex)
            self.rgbHexSV.set(hexColor)
        if not _PILLOW_INSTALLED or sys.platform == 'darwin' or (not (0 <= x < width and 0 <= y < height)):
            self.colorFrame.configure(background='black')
        else:
            self.colorFrame.configure(background=hexColor)
        if self.isRunning:
            self._updateMouseInfoJob = self.root.after(100, self._updateMouseInfoTextFields)
        else:
            return

    def _copyText(self, textToCopy):
        try:
            pyperclip.copy(textToCopy)
            self.statusbarSV.set('Copied ' + textToCopy)
        except pyperclip.PyperclipException as e:
            if platform.system() == 'Linux':
                self.statusbarSV.set('Copy failed. Run "sudo apt-get install xsel".')
            else:
                self.statusbarSV.set('Clipboard error: ' + str(e))

    def _copyXyMouseInfo(self, *args):
        if len(args) > 0 and isinstance(args[0], Event):
            args = ()
        if self.delayEnabledSV.get() == 'on' and len(args) == 0:
            self.root.after(1000, self._copyXyMouseInfo, 2)
            self.xyCopyButtonSV.set('Copy in 3')
        elif len(args) == 1 and args[0] == 2:
            self.root.after(1000, self._copyXyMouseInfo, 1)
            self.xyCopyButtonSV.set('Copy in 2')
        elif len(args) == 1 and args[0] == 1:
            self.root.after(1000, self._copyXyMouseInfo, 0)
            self.xyCopyButtonSV.set('Copy in 1')
        else:
            self._copyText(self.xyTextboxSV.get())
            self.xyCopyButtonSV.set('Copy XY')

    def _copyRgbMouseInfo(self, *args):
        if len(args) > 0 and isinstance(args[0], Event):
            args = ()
        if self.delayEnabledSV.get() == 'on' and len(args) == 0:
            self.root.after(1000, self._copyRgbMouseInfo, 2)
            self.rgbCopyButtonSV.set('Copy in 3')
        elif len(args) == 1 and args[0] == 2:
            self.root.after(1000, self._copyRgbMouseInfo, 1)
            self.rgbCopyButtonSV.set('Copy in 2')
        elif len(args) == 1 and args[0] == 1:
            self.root.after(1000, self._copyRgbMouseInfo, 0)
            self.rgbCopyButtonSV.set('Copy in 1')
        else:
            self._copyText(self.rgbSV.get())
            self.rgbCopyButtonSV.set('Copy RGB')

    def _copyRgbHexMouseInfo(self, *args):
        if len(args) > 0 and isinstance(args[0], Event):
            args = ()
        if self.delayEnabledSV.get() == 'on' and len(args) == 0:
            self.root.after(1000, self._copyRgbHexMouseInfo, 2)
            self.rgbHexCopyButtonSV.set('Copy in 3')
        elif len(args) == 1 and args[0] == 2:
            self.root.after(1000, self._copyRgbHexMouseInfo, 1)
            self.rgbHexCopyButtonSV.set('Copy in 2')
        elif len(args) == 1 and args[0] == 1:
            self.root.after(1000, self._copyRgbHexMouseInfo, 0)
            self.rgbHexCopyButtonSV.set('Copy in 1')
        else:
            self._copyText(self.rgbHexSV.get())
            self.rgbHexCopyButtonSV.set('Copy RGB Hex')

    def _copyAllMouseInfo(self, *args):
        textFieldContents = '%s %s %s' % (self.xyTextboxSV.get(), self.rgbSV.get(), self.rgbHexSV.get())
        if len(args) > 0 and isinstance(args[0], Event):
            args = ()
        if self.delayEnabledSV.get() == 'on' and len(args) == 0:
            self.root.after(1000, self._copyAllMouseInfo, 2)
            self.allCopyButtonSV.set('Copy in 3')
        elif len(args) == 1 and args[0] == 2:
            self.root.after(1000, self._copyAllMouseInfo, 1)
            self.allCopyButtonSV.set('Copy in 2')
        elif len(args) == 1 and args[0] == 1:
            self.root.after(1000, self._copyAllMouseInfo, 0)
            self.allCopyButtonSV.set('Copy in 1')
        else:
            textFieldContents = '%s %s %s' % (self.xyTextboxSV.get(), self.rgbSV.get(), self.rgbHexSV.get())
            self._copyText(textFieldContents)
            self.allCopyButtonSV.set('Copy All')

    def _logXyMouseInfo(self, *args):
        if len(args) > 0 and isinstance(args[0], Event):
            args = ()
        if self.delayEnabledSV.get() == 'on' and len(args) == 0:
            self.root.after(1000, self._logXyMouseInfo, 2)
            self.xyLogButtonSV.set('Log in 3')
        elif len(args) == 1 and args[0] == 2:
            self.root.after(1000, self._logXyMouseInfo, 1)
            self.xyLogButtonSV.set('Log in 2')
        elif len(args) == 1 and args[0] == 1:
            self.root.after(1000, self._logXyMouseInfo, 0)
            self.xyLogButtonSV.set('Log in 1')
        else:
            logContents = self.logTextarea.get('1.0', 'end-1c') + '%s\n' % self.xyTextboxSV.get()
            self.logTextboxSV.set(logContents)
            self._setLogTextAreaContents(logContents)
            self.statusbarSV.set('Logged ' + self.xyTextboxSV.get())
            self.xyLogButtonSV.set('Log XY')

    def _logRgbMouseInfo(self, *args):
        if len(args) > 0 and isinstance(args[0], Event):
            args = ()
        if self.delayEnabledSV.get() == 'on' and len(args) == 0:
            self.root.after(1000, self._logRgbMouseInfo, 2)
            self.rgbLogButtonSV.set('Log in 3')
        elif len(args) == 1 and args[0] == 2:
            self.root.after(1000, self._logRgbMouseInfo, 1)
            self.rgbLogButtonSV.set('Log in 2')
        elif len(args) == 1 and args[0] == 1:
            self.root.after(1000, self._logRgbMouseInfo, 0)
            self.rgbLogButtonSV.set('Log in 1')
        else:
            logContents = self.logTextarea.get('1.0', 'end-1c') + '%s\n' % self.rgbSV.get()
            self.logTextboxSV.set(logContents)
            self._setLogTextAreaContents(logContents)
            self.statusbarSV.set('Logged ' + self.rgbSV.get())
            self.rgbLogButtonSV.set('Log RGB')

    def _logRgbHexMouseInfo(self, *args):
        if len(args) > 0 and isinstance(args[0], Event):
            args = ()
        if self.delayEnabledSV.get() == 'on' and len(args) == 0:
            self.root.after(1000, self._logRgbHexMouseInfo, 2)
            self.rgbHexLogButtonSV.set('Log in 3')
        elif len(args) == 1 and args[0] == 2:
            self.root.after(1000, self._logRgbHexMouseInfo, 1)
            self.rgbHexLogButtonSV.set('Log in 2')
        elif len(args) == 1 and args[0] == 1:
            self.root.after(1000, self._logRgbHexMouseInfo, 0)
            self.rgbHexLogButtonSV.set('Log in 1')
        else:
            logContents = self.logTextarea.get('1.0', 'end-1c') + '%s\n' % self.rgbHexSV.get()
            self.logTextboxSV.set(logContents)
            self._setLogTextAreaContents(logContents)
            self.statusbarSV.set('Logged ' + self.rgbHexSV.get())
            self.rgbHexLogButtonSV.set('Log RGB Hex')

    def _logAllMouseInfo(self, *args):
        if len(args) > 0 and isinstance(args[0], Event):
            args = ()
        if self.delayEnabledSV.get() == 'on' and len(args) == 0:
            self.root.after(1000, self._logAllMouseInfo, 2)
            self.allLogButtonSV.set('Log in 3')
        elif len(args) == 1 and args[0] == 2:
            self.root.after(1000, self._logAllMouseInfo, 1)
            self.allLogButtonSV.set('Log in 2')
        elif len(args) == 1 and args[0] == 1:
            self.root.after(1000, self._logAllMouseInfo, 0)
            self.allLogButtonSV.set('Log in 1')
        else:
            textFieldContents = '%s %s %s' % (self.xyTextboxSV.get(), self.rgbSV.get(), self.rgbHexSV.get())
            logContents = self.logTextarea.get('1.0', 'end-1c') + '%s\n' % textFieldContents
            self.logTextboxSV.set(logContents)
            self._setLogTextAreaContents(logContents)
            self.statusbarSV.set('Logged ' + textFieldContents)
            self.allLogButtonSV.set('Log All')

    def _xyOriginChanged(self, sv):
        contents = sv.get()
        if len(contents.split(',')) != 2:
            return
        x, y = contents.split(',')
        x = x.strip()
        y = y.strip()
        if not x.isdecimal() or not y.isdecimal():
            return
        self.xOrigin = int(x)
        self.yOrigin = int(y)
        self.statusbarSV.set('Set XY Origin to ' + str(self.xOrigin) + ', ' + str(self.yOrigin))

    def _setLogTextAreaContents(self, logContents):
        if RUNNING_PYTHON_2:
            self.logTextarea.delete('1.0', tkinter.END)
            self.logTextarea.insert(tkinter.END, logContents)
        else:
            self.logTextarea.replace('1.0', tkinter.END, logContents)
        topOfTextArea, bottomOfTextArea = self.logTextarea.yview()
        self.logTextarea.yview_moveto(bottomOfTextArea)

    def _saveLogFile(self, *args):
        try:
            with open(self.logFilenameSV.get(), 'w') as fo:
                fo.write(self.logTextboxSV.get())
        except Exception as e:
            self.statusbarSV.set('ERROR: ' + str(e))
        else:
            self.statusbarSV.set('Log file saved to ' + self.logFilenameSV.get())

    def _saveScreenshotFile(self, *args):
        if not _PILLOW_INSTALLED:
            self.statusbarSV.set('ERROR: NA_Pillow_unsupported')
            return
        try:
            screenshot(self.screenshotFilenameSV.get())
        except Exception as e:
            self.statusbarSV.set('ERROR: ' + str(e))
        else:
            self.statusbarSV.set('Screenshot file saved to ' + self.screenshotFilenameSV.get())

    def __init__(self):
        """Launches the MouseInfo window, which displays XY coordinate and RGB
        color information for the mouse's current position."""
        self.isRunning = True
        self.root = tkinter.Tk()
        self.root.title('MouseInfo ' + __version__)
        self.root.minsize(400, 100)
        if RUNNING_PYTHON_2:
            mainframe = tkinter.Frame(self.root)
        else:
            mainframe = ttk.Frame(self.root, padding='3 3 12 12')
        mainframe.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S))
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)
        CUR_ROW = 1
        self.delayEnabledSV = tkinter.StringVar()
        self.delayEnabledSV.set('on')
        delayCheckbox = ttk.Checkbutton(mainframe, text='3 Sec. Button Delay', variable=self.delayEnabledSV, onvalue='on', offvalue='off')
        delayCheckbox.grid(column=1, row=CUR_ROW, columnspan=2, sticky=tkinter.W)
        self.allCopyButtonSV = tkinter.StringVar()
        self.allCopyButtonSV.set('Copy All (F1)')
        self.allCopyButton = ttk.Button(mainframe, textvariable=self.allCopyButtonSV, width=MOUSE_INFO_BUTTON_WIDTH, command=self._copyAllMouseInfo)
        self.allCopyButton.grid(column=3, row=CUR_ROW, sticky=tkinter.W)
        self.allCopyButton.bind('<Return>', self._copyAllMouseInfo)
        self.allLogButtonSV = tkinter.StringVar()
        self.allLogButtonSV.set('Log All (F5)')
        self.allLogButton = ttk.Button(mainframe, textvariable=self.allLogButtonSV, width=MOUSE_INFO_BUTTON_WIDTH, command=self._logAllMouseInfo)
        self.allLogButton.grid(column=4, row=CUR_ROW, sticky=tkinter.W)
        self.allLogButton.bind('<Return>', self._logAllMouseInfo)
        self.xyTextboxSV = tkinter.StringVar()
        self.rgbSV = tkinter.StringVar()
        self.rgbHexSV = tkinter.StringVar()
        self.xyOriginSV = tkinter.StringVar()
        self.logTextboxSV = tkinter.StringVar()
        self.logFilenameSV = tkinter.StringVar()
        self.screenshotFilenameSV = tkinter.StringVar()
        self.statusbarSV = tkinter.StringVar()
        CUR_ROW += 1
        self.xyInfoTextbox = ttk.Entry(mainframe, width=16, textvariable=self.xyTextboxSV)
        self.xyInfoTextbox.grid(column=2, row=CUR_ROW, sticky=(tkinter.W, tkinter.E))
        ttk.Label(mainframe, text='XY Position').grid(column=1, row=CUR_ROW, sticky=tkinter.W)
        self.xyCopyButtonSV = tkinter.StringVar()
        self.xyCopyButtonSV.set('Copy XY (F2)')
        self.xyCopyButton = ttk.Button(mainframe, textvariable=self.xyCopyButtonSV, width=MOUSE_INFO_BUTTON_WIDTH, command=self._copyXyMouseInfo)
        self.xyCopyButton.grid(column=3, row=CUR_ROW, sticky=tkinter.W)
        self.xyCopyButton.bind('<Return>', self._copyXyMouseInfo)
        self.xyLogButtonSV = tkinter.StringVar()
        self.xyLogButtonSV.set('Log XY (F6)')
        self.xyLogButton = ttk.Button(mainframe, textvariable=self.xyLogButtonSV, width=MOUSE_INFO_BUTTON_WIDTH, command=self._logXyMouseInfo)
        self.xyLogButton.grid(column=4, row=CUR_ROW, sticky=tkinter.W)
        self.xyLogButton.bind('<Return>', self._logXyMouseInfo)
        CUR_ROW += 1
        self.rgbSV_entry = ttk.Entry(mainframe, width=16, textvariable=self.rgbSV)
        self.rgbSV_entry.grid(column=2, row=CUR_ROW, sticky=(tkinter.W, tkinter.E))
        ttk.Label(mainframe, text='RGB Color').grid(column=1, row=CUR_ROW, sticky=tkinter.W)
        self.rgbCopyButtonSV = tkinter.StringVar()
        self.rgbCopyButtonSV.set('Copy RGB (F3)')
        self.rgbCopyButton = ttk.Button(mainframe, textvariable=self.rgbCopyButtonSV, width=MOUSE_INFO_BUTTON_WIDTH, command=self._copyRgbMouseInfo)
        self.rgbCopyButton.grid(column=3, row=CUR_ROW, sticky=tkinter.W)
        self.rgbCopyButton.bind('<Return>', self._copyRgbMouseInfo)
        self.rgbLogButtonSV = tkinter.StringVar()
        self.rgbLogButtonSV.set('Log RGB (F7)')
        self.rgbLogButton = ttk.Button(mainframe, textvariable=self.rgbLogButtonSV, width=MOUSE_INFO_BUTTON_WIDTH, command=self._logRgbMouseInfo)
        self.rgbLogButton.grid(column=4, row=CUR_ROW, sticky=tkinter.W)
        self.rgbLogButton.bind('<Return>', self._logRgbMouseInfo)
        CUR_ROW += 1
        self.rgbHexSV_entry = ttk.Entry(mainframe, width=16, textvariable=self.rgbHexSV)
        self.rgbHexSV_entry.grid(column=2, row=CUR_ROW, sticky=(tkinter.W, tkinter.E))
        ttk.Label(mainframe, text='RGB as Hex').grid(column=1, row=CUR_ROW, sticky=tkinter.W)
        self.rgbHexCopyButtonSV = tkinter.StringVar()
        self.rgbHexCopyButtonSV.set('Copy RGB Hex (F4)')
        self.rgbHexCopyButton = ttk.Button(mainframe, textvariable=self.rgbHexCopyButtonSV, width=MOUSE_INFO_BUTTON_WIDTH, command=self._copyRgbHexMouseInfo)
        self.rgbHexCopyButton.grid(column=3, row=CUR_ROW, sticky=tkinter.W)
        self.rgbHexCopyButton.bind('<Return>', self._copyRgbHexMouseInfo)
        self.rgbHexLogButtonSV = tkinter.StringVar()
        self.rgbHexLogButtonSV.set('Log RGB Hex (F8)')
        self.rgbHexLogButton = ttk.Button(mainframe, textvariable=self.rgbHexLogButtonSV, width=MOUSE_INFO_BUTTON_WIDTH, command=self._logRgbHexMouseInfo)
        self.rgbHexLogButton.grid(column=4, row=CUR_ROW, sticky=tkinter.W)
        self.rgbHexLogButton.bind('<Return>', self._logRgbHexMouseInfo)
        CUR_ROW += 1
        self.colorFrame = tkinter.Frame(mainframe, width=50, height=50)
        self.colorFrame.grid(column=2, row=CUR_ROW, sticky=(tkinter.W, tkinter.E))
        ttk.Label(mainframe, text='Color').grid(column=1, row=CUR_ROW, sticky=tkinter.W)
        CUR_ROW += 1
        self.xOrigin = 0
        self.yOrigin = 0
        self.xyOriginSV.set('0, 0')
        ttk.Label(mainframe, text='XY Origin').grid(column=1, row=CUR_ROW, sticky=tkinter.W)
        self.xyOriginSV.trace('w', lambda name, index, mode, sv=self.xyOriginSV: self._xyOriginChanged(sv))
        self.xyOriginSV_entry = ttk.Entry(mainframe, width=16, textvariable=self.xyOriginSV)
        self.xyOriginSV_entry.grid(column=2, row=CUR_ROW, sticky=(tkinter.W, tkinter.E))
        CUR_ROW += 1
        self.logTextarea = tkinter.Text(mainframe, width=20, height=6)
        self.logTextarea.grid(column=1, row=CUR_ROW, columnspan=4, sticky=(tkinter.W, tkinter.E, tkinter.N, tkinter.S))
        self.logTextareaScrollbar = ttk.Scrollbar(mainframe, orient=tkinter.VERTICAL, command=self.logTextarea.yview)
        self.logTextareaScrollbar.grid(column=5, row=CUR_ROW, sticky=(tkinter.N, tkinter.S))
        self.logTextarea['yscrollcommand'] = self.logTextareaScrollbar.set
        CUR_ROW += 1
        self.logFilenameTextbox = ttk.Entry(mainframe, width=16, textvariable=self.logFilenameSV)
        self.logFilenameTextbox.grid(column=1, row=CUR_ROW, columnspan=3, sticky=(tkinter.W, tkinter.E))
        self.saveLogButton = ttk.Button(mainframe, text='Save Log', width=MOUSE_INFO_BUTTON_WIDTH, command=self._saveLogFile)
        self.saveLogButton.grid(column=4, row=CUR_ROW, sticky=tkinter.W)
        self.saveLogButton.bind('<Return>', self._saveLogFile)
        self.logFilenameSV.set(os.path.join(os.getcwd(), 'mouseInfoLog.txt'))
        CUR_ROW += 1
        G_MOUSE_INFO_SCREENSHOT_FILENAME_entry = ttk.Entry(mainframe, width=16, textvariable=self.screenshotFilenameSV)
        G_MOUSE_INFO_SCREENSHOT_FILENAME_entry.grid(column=1, row=CUR_ROW, columnspan=3, sticky=(tkinter.W, tkinter.E))
        self.saveScreenshotButton = ttk.Button(mainframe, text='Save Screenshot', width=MOUSE_INFO_BUTTON_WIDTH, command=self._saveScreenshotFile)
        self.saveScreenshotButton.grid(column=4, row=CUR_ROW, sticky=tkinter.W)
        self.saveScreenshotButton.bind('<Return>', self._saveScreenshotFile)
        self.screenshotFilenameSV.set(os.path.join(os.getcwd(), 'mouseInfoScreenshot.png'))
        CUR_ROW += 1
        statusbar = ttk.Label(mainframe, relief=tkinter.SUNKEN, textvariable=self.statusbarSV)
        statusbar.grid(column=1, row=CUR_ROW, columnspan=5, sticky=(tkinter.W, tkinter.E))
        for child in mainframe.winfo_children():
            if child == self.logTextareaScrollbar:
                child.grid_configure(padx=0, pady=3)
            elif child == self.logTextarea:
                child.grid_configure(padx=(3, 0), pady=3)
            elif child == statusbar:
                child.grid_configure(padx=0, pady=(3, 0))
            else:
                child.grid_configure(padx=3, pady=3)
        self.root.option_add('*tearOff', tkinter.FALSE)
        menu = tkinter.Menu(self.root)
        self.root.config(menu=menu)
        copyMenu = tkinter.Menu(menu)
        copyMenu.add_command(label='Copy All', command=self._copyAllMouseInfo, accelerator='F1', underline=5)
        copyMenu.add_command(label='Copy XY', command=self._copyXyMouseInfo, accelerator='F2', underline=5)
        copyMenu.add_command(label='Copy RGB', command=self._copyRgbMouseInfo, accelerator='F3', underline=5)
        copyMenu.add_command(label='Copy RGB as Hex', command=self._copyRgbHexMouseInfo, accelerator='F4', underline=12)
        menu.add_cascade(label='Copy', menu=copyMenu, underline=0)
        logMenu = tkinter.Menu(menu)
        logMenu.add_command(label='Log All', command=self._logAllMouseInfo, accelerator='F5', underline=4)
        logMenu.add_command(label='Log XY', command=self._logXyMouseInfo, accelerator='F6', underline=4)
        logMenu.add_command(label='Log RGB', command=self._logRgbMouseInfo, accelerator='F7', underline=4)
        logMenu.add_command(label='Log RGB as Hex', command=self._logRgbHexMouseInfo, accelerator='F8', underline=11)
        menu.add_cascade(label='Log', menu=logMenu, underline=0)
        helpMenu = tkinter.Menu(menu)
        helpMenu.add_command(label='Online Documentation', command=lambda: webbrowser.open('https://mouseinfo.readthedocs.io'), underline=6)
        menu.add_cascade(label='Help', menu=helpMenu, underline=0)
        self.root.bind_all('<F1>', self._copyAllMouseInfo)
        self.root.bind_all('<F2>', self._copyXyMouseInfo)
        self.root.bind_all('<F3>', self._copyRgbMouseInfo)
        self.root.bind_all('<F4>', self._copyRgbHexMouseInfo)
        self.root.bind_all('<F5>', self._logAllMouseInfo)
        self.root.bind_all('<F6>', self._logXyMouseInfo)
        self.root.bind_all('<F7>', self._logRgbMouseInfo)
        self.root.bind_all('<F8>', self._logRgbHexMouseInfo)
        self.root.resizable(False, False)
        self.xyInfoTextbox.focus()
        self._updateMouseInfoJob = self.root.after(100, self._updateMouseInfoTextFields)
        self.root.attributes('-topmost', True)
        self.root.update()
        self.root.mainloop()
        self.root.after_cancel(self._updateMouseInfoJob)
        self.isRunning = False
        try:
            self.root.destroy()
        except tkinter.TclError:
            pass