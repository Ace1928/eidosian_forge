import time
import threading
class IPythonTkRoot(Tk):
    """
    A Tk root window intended for use in an IPython shell.

    Because of the way that IPython overloads the Python inputhook, it
    is necessary to start a Tk event loop by running the magic command
    %gui tk in order for Tk windows to actually appear on the screen.
    An IPythonRoot detects whether there is a Tk event loop
    running and, if not, reminds the user to type %gui tk.
    """

    def __init__(self, **kwargs):
        window_type = kwargs.pop('window_type', '')
        Tk.__init__(self, **kwargs)
        self.message = '\x1b[31mYour new {} window needs an event loop to become visible.\nType "%gui tk" below (without the quotes) to start one.\x1b[0m\n'.format(window_type if window_type else self.winfo_class())
        if ip and IPython.version_info < (6,):
            self.message = '\n' + self.message[:-1]
        self._have_loop = False
        self._check_for_tk()

    def _tk_check(self):
        for n in range(4):
            time.sleep(0.25)
            if self._have_loop:
                return
        print(self.message)

    def _check_for_tk(self):

        def set_flag():
            self._have_loop = True
        if ip:
            self.after(10, set_flag)
            threading.Thread(target=self._tk_check).start()