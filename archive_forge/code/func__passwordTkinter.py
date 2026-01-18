import sys
def _passwordTkinter(text='', title='', default='', mask='*', root=None, timeout=None):
    """Displays a message box with text input, and OK & Cancel buttons. Typed characters appear as *. Returns the text entered, or None if Cancel was clicked."""
    assert TKINTER_IMPORT_SUCCEEDED, 'Tkinter is required for pymsgbox'
    text = str(text)
    return __fillablebox(text, title, default, mask=mask, root=root, timeout=timeout)