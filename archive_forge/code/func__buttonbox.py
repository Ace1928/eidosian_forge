import sys
def _buttonbox(msg, title, choices, root=None, timeout=None):
    """
    Display a msg, a title, and a set of buttons.
    The buttons are defined by the members of the choices list.
    Return the text of the button that the user selected.

    @arg msg: the msg to be displayed.
    @arg title: the window title
    @arg choices: a list or tuple of the choices to be displayed
    """
    global boxRoot, __replyButtonText, __widgetTexts, buttonsFrame
    __replyButtonText = choices[0]
    if root:
        root.withdraw()
        boxRoot = tk.Toplevel(master=root)
        boxRoot.withdraw()
    else:
        boxRoot = tk.Tk()
        boxRoot.withdraw()
    boxRoot.title(title)
    boxRoot.iconname('Dialog')
    boxRoot.geometry(rootWindowPosition)
    boxRoot.minsize(400, 100)
    messageFrame = tk.Frame(master=boxRoot)
    messageFrame.pack(side=tk.TOP, fill=tk.BOTH)
    buttonsFrame = tk.Frame(master=boxRoot)
    buttonsFrame.pack(side=tk.TOP, fill=tk.BOTH)
    messageWidget = tk.Message(messageFrame, text=msg, width=400)
    messageWidget.configure(font=(PROPORTIONAL_FONT_FAMILY, PROPORTIONAL_FONT_SIZE))
    messageWidget.pack(side=tk.TOP, expand=tk.YES, fill=tk.X, padx='3m', pady='3m')
    __put_buttons_in_buttonframe(choices)
    __firstWidget.focus_force()
    boxRoot.deiconify()
    if timeout is not None:
        boxRoot.after(timeout, timeoutBoxRoot)
    boxRoot.mainloop()
    try:
        boxRoot.destroy()
    except tk.TclError:
        if __replyButtonText != TIMEOUT_RETURN_VALUE:
            __replyButtonText = None
    if root:
        root.deiconify()
    return __replyButtonText