import sys
def __put_buttons_in_buttonframe(choices):
    """Put the buttons in the buttons frame"""
    global __widgetTexts, __firstWidget, buttonsFrame
    __firstWidget = None
    __widgetTexts = {}
    i = 0
    for buttonText in choices:
        tempButton = tk.Button(buttonsFrame, takefocus=1, text=buttonText)
        _bindArrows(tempButton)
        tempButton.pack(expand=tk.YES, side=tk.LEFT, padx='1m', pady='1m', ipadx='2m', ipady='1m')
        __widgetTexts[tempButton] = buttonText
        if i == 0:
            __firstWidget = tempButton
            i = 1
        commandButton = tempButton
        handler = __buttonEvent
        for selectionEvent in STANDARD_SELECTION_EVENTS:
            commandButton.bind('<%s>' % selectionEvent, handler)
        if CANCEL_TEXT in choices:
            commandButton.bind('<Escape>', __cancelButtonEvent)