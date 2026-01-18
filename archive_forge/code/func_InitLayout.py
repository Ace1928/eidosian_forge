import wx
def InitLayout(self):
    """Lay out controls and add buttons."""
    sizer = wx.BoxSizer(wx.HORIZONTAL)
    txtSizer = wx.BoxSizer(wx.VERTICAL)
    btnSizer = wx.BoxSizer(wx.VERTICAL)
    replaceSizer = wx.BoxSizer(wx.HORIZONTAL)
    txtSizer.Add(wx.StaticText(self, -1, 'Unrecognised Word:'), 0, wx.LEFT | wx.TOP, 5)
    txtSizer.Add(self.error_text, 1, wx.ALL | wx.EXPAND, 5)
    replaceSizer.Add(wx.StaticText(self, -1, 'Replace with:'), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
    replaceSizer.Add(self.replace_text, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
    txtSizer.Add(replaceSizer, 0, wx.EXPAND, 0)
    txtSizer.Add(self.replace_list, 2, wx.ALL | wx.EXPAND, 5)
    sizer.Add(txtSizer, 1, wx.EXPAND, 0)
    self.buttons = []
    for label, action, tip in (('Ignore', self.OnIgnore, 'Ignore this word and continue'), ('Ignore All', self.OnIgnoreAll, 'Ignore all instances of this word and continue'), ('Replace', self.OnReplace, 'Replace this word'), ('Replace All', self.OnReplaceAll, 'Replace all instances of this word'), ('Add', self.OnAdd, 'Add this word to the dictionary'), ('Done', self.OnDone, 'Finish spell-checking and accept changes')):
        btn = wx.Button(self, -1, label)
        btn.SetToolTip(wx.ToolTip(tip))
        btnSizer.Add(btn, 0, wx.ALIGN_RIGHT | wx.ALL, 4)
        btn.Bind(wx.EVT_BUTTON, action)
        self.buttons.append(btn)
    sizer.Add(btnSizer, 0, wx.ALL | wx.EXPAND, 5)
    self.SetAutoLayout(True)
    self.SetSizer(sizer)
    sizer.Fit(self)