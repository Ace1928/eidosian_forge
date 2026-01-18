from Xlib.X import NoSymbol
Translate a keysym (16 bit number) into a python string.

    This will pass 0 to 0xff as well as XK_BackSpace, XK_Tab, XK_Clear,
    XK_Return, XK_Pause, XK_Scroll_Lock, XK_Escape, XK_Delete. For other
    values it returns None.