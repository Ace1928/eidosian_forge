from jnius import autoclass, java_method, PythonJavaClass
from android import api_version
from kivy.core.audio import Sound, SoundLoader
class OnCompletionListener(PythonJavaClass):
    __javainterfaces__ = ['android/media/MediaPlayer$OnCompletionListener']
    __javacontext__ = 'app'

    def __init__(self, callback, **kwargs):
        super(OnCompletionListener, self).__init__(**kwargs)
        self.callback = callback

    @java_method('(Landroid/media/MediaPlayer;)V')
    def onCompletion(self, mp):
        self.callback()